import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet, CENTER, ROTATION_RADIUS_LIMIT

# ---------- constants ----------
MAX_SPEED   = 6.0
MIN_SPEED   = 1.0
SUN_X, SUN_Y = 50.0, 50.0
SUN_RADIUS  = 10.0
BOARD       = 100.0

ATTACK_BUFFER       = 3      # extra ships over garrison when attacking
MAX_LEAD_ITERS      = 6      # orbit intercept iterations
PRODUCTION_WEIGHT   = 10.0   # score multiplier for production
COMET_BONUS         = 2.0    # extra score multiplier for comets (free production)
GARRISON_FRAC       = 0.20   # keep this fraction at home
MIN_GARRISON        = 8      # absolute floor regardless of fraction
DEFEND_RATIO        = 1.15   # defend if enemy incoming > garrison * this ratio
FLEET_ANGLE_TOL     = 0.25   # radians — fleet "aimed at" tolerance
SURPLUS_THRESHOLD   = 80     # send surplus ships to richest planet if over this
MAX_ATTACKS_PER_TURN = 3     # cap moves per planet to avoid splitting too thin


# ---------- physics ----------

def fleet_speed(ships: int) -> float:
    if ships <= 1:
        return MIN_SPEED
    return MIN_SPEED + (MAX_SPEED - MIN_SPEED) * (math.log(ships) / math.log(1000)) ** 1.5


def dist(x1, y1, x2, y2) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def travel_time(x1, y1, x2, y2, ships: int) -> float:
    d = dist(x1, y1, x2, y2)
    if d < 1e-9:
        return 0.0
    return d / fleet_speed(max(1, ships))


def planet_position(p: Planet, turns_ahead: float, angular_velocity: float, initial_planets: dict) -> tuple:
    orbital_radius = dist(p.x, p.y, SUN_X, SUN_Y)
    if orbital_radius + p.radius < ROTATION_RADIUS_LIMIT:
        ip = initial_planets.get(p.id)
        if ip is None:
            return p.x, p.y
        r = dist(ip.x, ip.y, SUN_X, SUN_Y)
        current_angle = math.atan2(p.y - SUN_Y, p.x - SUN_X)
        future_angle  = current_angle + angular_velocity * turns_ahead
        return SUN_X + r * math.cos(future_angle), SUN_Y + r * math.sin(future_angle)
    return p.x, p.y


def aim_at_moving_target(src_x, src_y, ships, target: Planet, angular_velocity: float, initial_planets: dict):
    """Iterative intercept solution. Returns (angle, eta, tx, ty)."""
    tx, ty = target.x, target.y
    for _ in range(MAX_LEAD_ITERS):
        eta = travel_time(src_x, src_y, tx, ty, ships)
        tx, ty = planet_position(target, eta, angular_velocity, initial_planets)
    angle = math.atan2(ty - src_y, tx - src_x)
    return angle, eta, tx, ty


def path_crosses_sun(x1, y1, x2, y2) -> bool:
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - SUN_X, y1 - SUN_Y
    a = dx*dx + dy*dy
    if a < 1e-9:
        return False
    b = 2*(fx*dx + fy*dy)
    c = fx*fx + fy*fy - SUN_RADIUS*SUN_RADIUS
    disc = b*b - 4*a*c
    if disc < 0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2*a)
    t2 = (-b + sq) / (2*a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1)


def angle_diff(a, b) -> float:
    return abs((a - b + math.pi) % (2*math.pi) - math.pi)


# ---------- fleet tracking ----------

def fleets_aimed_at(planet: Planet, fleets: list, owner_filter) -> list:
    """Return fleets whose angle points within tolerance at planet and match owner_filter."""
    result = []
    for f in fleets:
        if not owner_filter(f.owner):
            continue
        expected = math.atan2(planet.y - f.y, planet.x - f.x)
        if angle_diff(f.angle, expected) < FLEET_ANGLE_TOL:
            result.append(f)
    return result


def enemy_incoming(planet: Planet, fleets: list, player: int) -> int:
    fs = fleets_aimed_at(planet, fleets, lambda o: o not in (-1, player))
    return sum(f.ships for f in fs)


def friendly_incoming(planet: Planet, fleets: list, player: int) -> int:
    fs = fleets_aimed_at(planet, fleets, lambda o: o == player)
    return sum(f.ships for f in fs)


# ---------- comet helpers ----------

def comet_eta(comet_planet: Planet, comet_data: list, planet_id_to_comet: dict) -> int:
    """Return turns until this comet leaves the board, or 999 if unknown."""
    group = planet_id_to_comet.get(comet_planet.id)
    if group is None:
        return 999
    path  = group.get("paths", [[]])[0]
    idx   = group.get("path_index", 0)
    return max(0, len(path) - idx)


# ---------- scoring ----------

def score_target(src: Planet, target: Planet, angular_velocity: float,
                 initial_planets: dict, comet_ids: set,
                 friendly_en_route: int, turn: int) -> float:
    """Score a potential attack target. Higher = better."""
    tx, ty = planet_position(target, 0, angular_velocity, initial_planets)
    d = dist(src.x, src.y, tx, ty)
    if d < 1e-6:
        return -999.0

    prod = target.production
    if target.id in comet_ids:
        prod *= COMET_BONUS          # comets give time-limited production bonus

    cost = max(1, target.ships + ATTACK_BUFFER - friendly_en_route)
    score = (prod * PRODUCTION_WEIGHT) / (d + cost)

    if path_crosses_sun(src.x, src.y, tx, ty):
        score *= 0.2                 # heavy penalty — fleet would be destroyed

    # Late game: devalue neutrals vs enemies (reducing their planets matters more)
    if turn > 300 and target.owner == -1:
        score *= 0.7

    return score


# ---------- garrison calculation ----------

def garrison_floor(planet: Planet, enemy_fleets_incoming: int) -> int:
    """How many ships must stay on this planet."""
    base = max(MIN_GARRISON, int(planet.ships * GARRISON_FRAC))
    # If under threat, keep enough to survive + buffer
    if enemy_fleets_incoming > 0:
        base = max(base, enemy_fleets_incoming + ATTACK_BUFFER)
    return base


# ---------- main agent ----------

def agent(obs):
    # ── parse obs ──────────────────────────────────────────────────────────
    if isinstance(obs, dict):
        player         = obs.get("player", 0)
        raw_planets    = obs.get("planets", [])
        raw_fleets     = obs.get("fleets", [])
        angular_vel    = obs.get("angular_velocity", 0.0)
        raw_initial    = obs.get("initial_planets", [])
        comet_ids      = set(obs.get("comet_planet_ids", []))
        raw_comets     = obs.get("comets", [])
        step           = obs.get("step", 0)
    else:
        player         = obs.player
        raw_planets    = obs.planets
        raw_fleets     = obs.fleets
        angular_vel    = obs.angular_velocity
        raw_initial    = obs.initial_planets
        comet_ids      = set(obs.comet_planet_ids)
        raw_comets     = getattr(obs, "comets", [])
        step           = getattr(obs, "step", 0)

    planets         = [Planet(*p) for p in raw_planets]
    fleets          = [Fleet(*f)  for f in raw_fleets]
    initial_planets = {p[0]: Planet(*p) for p in raw_initial}

    # Build comet group lookup: planet_id -> group dict
    planet_id_to_comet = {}
    for grp in (raw_comets if isinstance(raw_comets, list) else []):
        grp_dict = grp if isinstance(grp, dict) else {}
        for pid in grp_dict.get("planet_ids", []):
            planet_id_to_comet[pid] = grp_dict

    my_planets      = [p for p in planets if p.owner == player]
    enemy_planets   = [p for p in planets if p.owner not in (-1, player)]
    neutral_planets = [p for p in planets if p.owner == -1]
    all_targets     = neutral_planets + enemy_planets

    if not my_planets:
        return []

    # ── pre-compute threat & friendly en-route for all planets ────────────
    planet_threat    = {p.id: enemy_incoming(p, fleets, player)   for p in my_planets}
    friendly_en_route = {p.id: friendly_incoming(p, fleets, player) for p in planets}
    committed      = {p.id: 0 for p in my_planets}  # ships earmarked this turn
    moves          = []

    # ── PHASE 1: Emergency defense ─────────────────────────────────────────
    # Threatened planets get reinforced from closest neighbours.
    threatened = sorted(
        [p for p in my_planets if planet_threat[p.id] > p.ships * DEFEND_RATIO],
        key=lambda p: -(planet_threat[p.id] - p.ships)  # worst first
    )

    for t_planet in threatened:
        needed = int(planet_threat[t_planet.id] * DEFEND_RATIO) - t_planet.ships
        needed = max(needed, ATTACK_BUFFER)
        helpers = sorted(
            [p for p in my_planets if p.id != t_planet.id],
            key=lambda p: dist(p.x, p.y, t_planet.x, t_planet.y)
        )
        for helper in helpers:
            threat_on_helper = planet_threat.get(helper.id, 0)
            floor = garrison_floor(helper, threat_on_helper)
            sendable = (helper.ships - committed[helper.id]) - floor
            if sendable <= 0:
                continue
            to_send = min(sendable, needed)
            if to_send < 1:
                continue
            angle = math.atan2(t_planet.y - helper.y, t_planet.x - helper.x)
            moves.append([helper.id, angle, to_send])
            committed[helper.id] += to_send
            needed -= to_send
            if needed <= 0:
                break

    if not all_targets:
        return moves

    # ── PHASE 2: Comet interception ────────────────────────────────────────
    # Grab comets that are passing nearby and are cheap — before normal attack loop.
    comet_planets = [p for p in neutral_planets if p.id in comet_ids]
    for comet in comet_planets:
        turns_left = comet_eta(comet, [], planet_id_to_comet)
        if turns_left < 5:
            continue  # leaving too soon, not worth it

        # Find cheapest planet to launch from
        best_src = None
        best_eta = float("inf")
        for src in my_planets:
            _, eta, _, _ = aim_at_moving_target(src.x, src.y, comet.ships + ATTACK_BUFFER,
                                                 comet, angular_vel, initial_planets)
            if eta < turns_left * 0.8 and eta < best_eta:
                best_eta = eta
                best_src = src

        if best_src is None:
            continue

        threat_on_src = planet_threat.get(best_src.id, 0)
        floor    = garrison_floor(best_src, threat_on_src)
        sendable = (best_src.ships - committed[best_src.id]) - floor
        needed   = comet.ships + ATTACK_BUFFER  # overwritten below with precomputed
        f_en_route_comet = friendly_en_route.get(comet.id, 0)
        needed   = comet.ships + ATTACK_BUFFER - f_en_route_comet
        to_send  = min(sendable, max(0, needed))

        if to_send < 1:
            continue

        angle, _, tx, ty = aim_at_moving_target(best_src.x, best_src.y, to_send,
                                                  comet, angular_vel, initial_planets)
        if path_crosses_sun(best_src.x, best_src.y, tx, ty):
            continue

        moves.append([best_src.id, angle, to_send])
        committed[best_src.id] += to_send
        # Remove comet from normal target list so we don't double-attack
        all_targets = [t for t in all_targets if t.id != comet.id]

    # ── PHASE 3: Attack ────────────────────────────────────────────────────
    # Each planet independently ranks all targets and fires at the best affordable one.
    # Rich planets with surplus can fire at multiple targets per turn.

    for src in my_planets:
        threat_on_src = planet_threat.get(src.id, 0)
        floor    = garrison_floor(src, threat_on_src)
        sendable = (src.ships - committed[src.id]) - floor
        if sendable < 2:
            continue

        # Score every target
        scored = []
        for t in all_targets:
            f_en_route = friendly_en_route.get(t.id, 0)
            s = score_target(src, t, angular_vel, initial_planets, comet_ids, f_en_route, step)
            scored.append((s, t))
        scored.sort(key=lambda x: -x[0])

        attacks_this_planet = 0
        remaining_sendable  = sendable

        for _, best in scored:
            if attacks_this_planet >= MAX_ATTACKS_PER_TURN:
                break
            if remaining_sendable < 2:
                break

            angle, eta, tx, ty = aim_at_moving_target(
                src.x, src.y, remaining_sendable,
                best, angular_vel, initial_planets
            )

            if path_crosses_sun(src.x, src.y, tx, ty):
                continue

            # Estimate garrison at arrival
            f_en_route = friendly_en_route.get(best.id, 0)
            if best.owner == -1:
                garrison_arr = best.ships
            else:
                garrison_arr = best.ships + int(eta * best.production)

            needed = max(1, garrison_arr + ATTACK_BUFFER - f_en_route)

            # Only fire if we can afford to actually capture it, or if
            # we're weakening a strong enemy planet (war of attrition)
            if needed > remaining_sendable:
                # Attrition: send everything we have if it significantly hurts them
                if best.owner != -1 and remaining_sendable > best.ships * 0.5:
                    to_send = remaining_sendable
                else:
                    continue
            else:
                to_send = needed

            to_send = min(to_send, remaining_sendable)
            if to_send < 1:
                continue

            moves.append([src.id, angle, to_send])
            committed[src.id]  += to_send
            remaining_sendable -= to_send
            attacks_this_planet += 1

    # ── PHASE 4: Surplus redistribution ───────────────────────────────────
    # Planets that are overflowing ships (nothing to attack) funnel surplus
    # to our richest production planet so ships aren't wasted sitting idle.
    if my_planets:
        richest = max(my_planets, key=lambda p: p.production)
        for src in my_planets:
            if src.id == richest.id:
                continue
            threat_on_src = planet_threat.get(src.id, 0)
            floor    = garrison_floor(src, threat_on_src)
            surplus  = (src.ships - committed[src.id]) - floor
            if surplus >= SURPLUS_THRESHOLD:
                angle = math.atan2(richest.y - src.y, richest.x - src.x)
                if not path_crosses_sun(src.x, src.y, richest.x, richest.y):
                    moves.append([src.id, angle, surplus])
                    committed[src.id] += surplus

    return moves