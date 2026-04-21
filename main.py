import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet, CENTER, ROTATION_RADIUS_LIMIT

# ---------- constants ----------
MAX_SPEED    = 6.0
MIN_SPEED    = 1.0
SUN_X, SUN_Y = 50.0, 50.0
SUN_RADIUS   = 10.0
BOARD        = 100.0

ATTACK_BUFFER        = 3
MAX_LEAD_ITERS       = 6
PRODUCTION_WEIGHT    = 12.0
COMET_BONUS          = 2.5
GARRISON_FRAC        = 0.18
MIN_GARRISON         = 8
DEFEND_RATIO         = 1.15
FLEET_ANGLE_TOL      = 0.28
SURPLUS_THRESHOLD    = 70
MAX_ATTACKS_PER_TURN = 4
EARLY_RUSH_TURNS     = 40
MASS_FLEET_BONUS     = 1.3
MIN_MASS_FLEET       = 50
INTERCEPT_WINDOW     = 10     # tighter window — only intercept very close threats
INTERCEPT_MIN_SHIPS  = 30     # only intercept fleets large enough to matter


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


def planet_position(p: Planet, turns_ahead: float, angular_vel: float, initial_planets: dict) -> tuple:
    orbital_radius = dist(p.x, p.y, SUN_X, SUN_Y)
    if orbital_radius + p.radius < ROTATION_RADIUS_LIMIT:
        ip = initial_planets.get(p.id)
        if ip is None:
            return p.x, p.y
        r = dist(ip.x, ip.y, SUN_X, SUN_Y)
        current_angle = math.atan2(p.y - SUN_Y, p.x - SUN_X)
        future_angle  = current_angle + angular_vel * turns_ahead
        return SUN_X + r * math.cos(future_angle), SUN_Y + r * math.sin(future_angle)
    return p.x, p.y


def aim_at_moving_target(src_x, src_y, ships, target: Planet, angular_vel: float, initial_planets: dict):
    """Iterative intercept. Returns (angle, eta, tx, ty)."""
    tx, ty = target.x, target.y
    for _ in range(MAX_LEAD_ITERS):
        eta = travel_time(src_x, src_y, tx, ty, ships)
        tx, ty = planet_position(target, eta, angular_vel, initial_planets)
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


# ---------- fleet position projection ----------

def fleet_position_at(f: Fleet, turns: float) -> tuple:
    """Where will fleet f be in `turns` turns?"""
    speed = fleet_speed(f.ships)
    return f.x + math.cos(f.angle) * speed * turns, f.y + math.sin(f.angle) * speed * turns


# ---------- comet helpers ----------

def comet_turns_left(comet_planet: Planet, planet_id_to_comet: dict) -> int:
    group = planet_id_to_comet.get(comet_planet.id)
    if group is None:
        return 999
    path = group.get("paths", [[]])[0]
    idx  = group.get("path_index", 0)
    return max(0, len(path) - idx)


# ---------- garrison ----------

def garrison_floor(planet: Planet, enemy_threat: int, my_production_rank: float) -> int:
    """
    Higher-production planets get bigger garrisons.
    my_production_rank: 0=lowest producer, 1=highest producer among my planets.
    """
    base = max(MIN_GARRISON, int(planet.ships * (GARRISON_FRAC + 0.10 * my_production_rank)))
    if enemy_threat > 0:
        base = max(base, enemy_threat + ATTACK_BUFFER)
    return base


# ---------- scoring ----------

def score_target(src: Planet, target: Planet, angular_vel: float,
                 initial_planets: dict, comet_ids: set,
                 f_en_route: int, step: int,
                 sendable: int = 1) -> float:
    tx, ty = planet_position(target, 0, angular_vel, initial_planets)
    d = dist(src.x, src.y, tx, ty)
    if d < 1e-6:
        return -999.0

    prod = target.production
    if target.id in comet_ids:
        prod *= COMET_BONUS

    cost = max(1, target.ships + ATTACK_BUFFER - f_en_route)
    score = (prod * PRODUCTION_WEIGHT) / (d + cost)

    # Bonus for massed fleets — they travel faster, arrive sooner
    if sendable >= MIN_MASS_FLEET:
        score *= MASS_FLEET_BONUS

    if path_crosses_sun(src.x, src.y, tx, ty):
        score *= 0.15

    # Early game: value neutrals highly to grow fast
    if step < EARLY_RUSH_TURNS and target.owner == -1:
        score *= 1.5

    # Late game: focus on eliminating enemies
    if step > 300 and target.owner == -1:
        score *= 0.6

    return score


# ---------- enemy fleet interception ----------

def find_intercepts(my_planets: list, enemy_fleets: list,
                    angular_vel: float, initial_planets: dict,
                    committed: dict, planet_threat: dict) -> list:
    """
    Detect enemy fleets passing within striking range of our planets.
    Returns list of (src_planet, intercept_angle, ships_to_send).
    """
    intercepts = []
    for ef in enemy_fleets:
        if ef.ships < INTERCEPT_MIN_SHIPS:
            continue
        # Project fleet position for next INTERCEPT_WINDOW turns
        speed = fleet_speed(ef.ships)
        for t in range(1, INTERCEPT_WINDOW + 1):
            fx = ef.x + math.cos(ef.angle) * speed * t
            fy = ef.y + math.sin(ef.angle) * speed * t
            # Out of bounds — fleet will die anyway
            if not (0 <= fx <= BOARD and 0 <= fy <= BOARD):
                break
            # Check if any of our planets can reach this intercept point in t turns
            for src in my_planets:
                tt = travel_time(src.x, src.y, fx, fy, ef.ships + ATTACK_BUFFER)
                if tt > t + 2:
                    continue
                # Can we get there in time?
                threat = planet_threat.get(src.id, 0)
                # Simple garrison: keep enough to survive own threats
                floor = max(MIN_GARRISON, threat + ATTACK_BUFFER)
                available = src.ships - committed.get(src.id, 0) - floor
                to_send = min(available, ef.ships + ATTACK_BUFFER)
                if to_send < ef.ships:
                    continue  # not enough to win the interception
                # Only intercept if path doesn't cross sun
                if path_crosses_sun(src.x, src.y, fx, fy):
                    continue
                angle = math.atan2(fy - src.y, fx - src.x)
                intercepts.append((src, angle, to_send, ef.ships))
                break  # one intercept opportunity per enemy fleet
            else:
                continue
            break
    return intercepts


# ---------- coordinated attack planning ----------

def plan_coordinated_attacks(hard_targets: list, my_planets: list,
                              angular_vel: float, initial_planets: dict,
                              committed: dict, planet_threat: dict,
                              friendly_en_route: dict, prod_rank: dict) -> list:
    """
    For heavily defended targets, coordinate multiple planets to attack together.
    Returns moves list.
    """
    moves = []
    for target in hard_targets:
        needed = target.ships + ATTACK_BUFFER - friendly_en_route.get(target.id, 0)
        if needed <= 0:
            continue
        # Sort planets by distance to target
        contributors = sorted(my_planets,
                              key=lambda p: dist(p.x, p.y, target.x, target.y))
        total_sent = 0
        for src in contributors:
            if total_sent >= needed:
                break
            rank = prod_rank.get(src.id, 0.0)
            floor = garrison_floor(src, planet_threat.get(src.id, 0), rank)
            available = src.ships - committed.get(src.id, 0) - floor
            if available < 2:
                continue
            to_send = min(available, needed - total_sent)
            angle, _, tx, ty = aim_at_moving_target(src.x, src.y, to_send,
                                                     target, angular_vel, initial_planets)
            if path_crosses_sun(src.x, src.y, tx, ty):
                continue
            moves.append([src.id, angle, to_send])
            committed[src.id] = committed.get(src.id, 0) + to_send
            total_sent += to_send
    return moves


# ---------- main agent ----------

def agent(obs):
    # ── parse obs ──────────────────────────────────────────────────────────
    if isinstance(obs, dict):
        player       = obs.get("player", 0)
        raw_planets  = obs.get("planets", [])
        raw_fleets   = obs.get("fleets", [])
        angular_vel  = obs.get("angular_velocity", 0.0)
        raw_initial  = obs.get("initial_planets", [])
        comet_ids    = set(obs.get("comet_planet_ids", []))
        raw_comets   = obs.get("comets", [])
        step         = obs.get("step", 0)
    else:
        player       = obs.player
        raw_planets  = obs.planets
        raw_fleets   = obs.fleets
        angular_vel  = obs.angular_velocity
        raw_initial  = obs.initial_planets
        comet_ids    = set(obs.comet_planet_ids)
        raw_comets   = getattr(obs, "comets", [])
        step         = getattr(obs, "step", 0)

    planets         = [Planet(*p) for p in raw_planets]
    fleets          = [Fleet(*f)  for f in raw_fleets]
    initial_planets = {p[0]: Planet(*p) for p in raw_initial}

    planet_id_to_comet = {}
    for grp in (raw_comets if isinstance(raw_comets, list) else []):
        g = grp if isinstance(grp, dict) else {}
        for pid in g.get("planet_ids", []):
            planet_id_to_comet[pid] = g

    my_planets      = [p for p in planets if p.owner == player]
    enemy_planets   = [p for p in planets if p.owner not in (-1, player)]
    neutral_planets = [p for p in planets if p.owner == -1]
    all_targets     = neutral_planets + enemy_planets

    if not my_planets:
        return []

    # ── production rank (0=lowest, 1=highest among my planets) ────────────
    if len(my_planets) > 1:
        min_prod = min(p.production for p in my_planets)
        max_prod = max(p.production for p in my_planets)
        prod_range = max(1, max_prod - min_prod)
        prod_rank = {p.id: (p.production - min_prod) / prod_range for p in my_planets}
    else:
        prod_rank = {my_planets[0].id: 1.0}

    # ── pre-compute threats & friendly en-route ────────────────────────────
    planet_threat     = {p.id: enemy_incoming(p, fleets, player) for p in my_planets}
    friendly_en_route = {p.id: friendly_incoming(p, fleets, player) for p in planets}
    committed         = {p.id: 0 for p in my_planets}
    moves             = []

    enemy_fleets = [f for f in fleets if f.owner not in (-1, player)]

    # ── PHASE 1: Emergency defense ─────────────────────────────────────────
    threatened = sorted(
        [p for p in my_planets if planet_threat[p.id] > p.ships * DEFEND_RATIO],
        key=lambda p: -(planet_threat[p.id] - p.ships)
    )
    for t_planet in threatened:
        needed = int(planet_threat[t_planet.id] * DEFEND_RATIO) - t_planet.ships
        needed = max(needed, ATTACK_BUFFER)
        helpers = sorted(
            [p for p in my_planets if p.id != t_planet.id],
            key=lambda p: dist(p.x, p.y, t_planet.x, t_planet.y)
        )
        for helper in helpers:
            rank  = prod_rank.get(helper.id, 0.0)
            floor = garrison_floor(helper, planet_threat.get(helper.id, 0), rank)
            sendable = (helper.ships - committed[helper.id]) - floor
            if sendable < 1:
                continue
            to_send = min(sendable, needed)
            angle = math.atan2(t_planet.y - helper.y, t_planet.x - helper.x)
            moves.append([helper.id, angle, to_send])
            committed[helper.id] += to_send
            needed -= to_send
            if needed <= 0:
                break

    if not all_targets:
        return moves

    # ── PHASE 2: Enemy fleet interception ─────────────────────────────────
    # Only intercept fleets that are actually aimed at one of our planets
    relevant_enemy_fleets = [
        f for f in enemy_fleets
        if any(angle_diff(f.angle, math.atan2(p.y - f.y, p.x - f.x)) < FLEET_ANGLE_TOL
               for p in my_planets)
    ]
    intercepts = find_intercepts(my_planets, relevant_enemy_fleets, angular_vel,
                                 initial_planets, committed, planet_threat)
    for src, angle, to_send, enemy_size in intercepts:
        # Double-check budget
        rank  = prod_rank.get(src.id, 0.0)
        floor = garrison_floor(src, planet_threat.get(src.id, 0), rank)
        available = src.ships - committed[src.id] - floor
        actual_send = min(available, to_send)
        if actual_send >= enemy_size:
            moves.append([src.id, angle, actual_send])
            committed[src.id] += actual_send

    # ── PHASE 3: Comet interception ────────────────────────────────────────
    comet_planets = [p for p in neutral_planets if p.id in comet_ids]
    for comet in sorted(comet_planets, key=lambda c: -c.production):
        turns_left = comet_turns_left(comet, planet_id_to_comet)
        if turns_left < 6:
            continue

        best_src, best_eta = None, float("inf")
        for src in my_planets:
            _, eta, _, _ = aim_at_moving_target(src.x, src.y,
                                                 comet.ships + ATTACK_BUFFER,
                                                 comet, angular_vel, initial_planets)
            if eta < turns_left * 0.75 and eta < best_eta:
                best_eta, best_src = eta, src

        if best_src is None:
            continue

        rank     = prod_rank.get(best_src.id, 0.0)
        floor    = garrison_floor(best_src, planet_threat.get(best_src.id, 0), rank)
        sendable = (best_src.ships - committed[best_src.id]) - floor
        needed   = max(0, comet.ships + ATTACK_BUFFER - friendly_en_route.get(comet.id, 0))
        to_send  = min(sendable, needed)
        if to_send < 1:
            continue

        angle, _, tx, ty = aim_at_moving_target(best_src.x, best_src.y, to_send,
                                                  comet, angular_vel, initial_planets)
        if path_crosses_sun(best_src.x, best_src.y, tx, ty):
            continue

        moves.append([best_src.id, angle, to_send])
        committed[best_src.id] += to_send
        all_targets = [t for t in all_targets if t.id != comet.id]

    # ── PHASE 4: Coordinated assault on hard targets ───────────────────────
    # Identify enemy planets that require more ships than any single planet can provide.
    hard_targets = [
        t for t in enemy_planets
        if t.ships + ATTACK_BUFFER - friendly_en_route.get(t.id, 0) >
           max((p.ships - committed[p.id] for p in my_planets), default=0)
        and t.production >= 3
    ]
    coord_moves = plan_coordinated_attacks(
        hard_targets, my_planets, angular_vel, initial_planets,
        committed, planet_threat, friendly_en_route, prod_rank
    )
    moves.extend(coord_moves)

    # ── PHASE 5: Individual attacks ────────────────────────────────────────
    # Remaining planets independently pick best target.
    remaining_targets = [t for t in all_targets if t.id not in {t.id for t in hard_targets}]

    for src in sorted(my_planets, key=lambda p: -p.ships):  # richest first
        rank     = prod_rank.get(src.id, 0.0)
        floor    = garrison_floor(src, planet_threat.get(src.id, 0), rank)
        sendable = (src.ships - committed[src.id]) - floor
        if sendable < 2:
            continue

        scored = []
        for t in remaining_targets:
            f_er = friendly_en_route.get(t.id, 0)
            s = score_target(src, t, angular_vel, initial_planets,
                             comet_ids, f_er, step, sendable)
            scored.append((s, t))
        scored.sort(key=lambda x: -x[0])

        attacks   = 0
        remaining = sendable

        for _, best in scored:
            if attacks >= MAX_ATTACKS_PER_TURN or remaining < 2:
                break

            angle, eta, tx, ty = aim_at_moving_target(
                src.x, src.y, remaining, best, angular_vel, initial_planets)

            if path_crosses_sun(src.x, src.y, tx, ty):
                continue

            f_er = friendly_en_route.get(best.id, 0)
            garrison_arr = best.ships if best.owner == -1 else best.ships + int(eta * best.production)
            needed = max(1, garrison_arr + ATTACK_BUFFER - f_er)

            # Early rush: attack even if slightly underpowered
            underpowered = needed > remaining
            if underpowered:
                if step < EARLY_RUSH_TURNS and best.owner == -1 and remaining >= best.ships:
                    to_send = remaining  # rush it
                elif best.owner != -1 and remaining > best.ships * 0.5:
                    to_send = remaining  # attrition
                else:
                    continue
            else:
                to_send = needed

            to_send = min(to_send, remaining)
            if to_send < 1:
                continue

            moves.append([src.id, angle, to_send])
            committed[src.id] += to_send
            remaining -= to_send
            attacks   += 1

    # ── PHASE 6: Surplus redistribution ───────────────────────────────────
    if my_planets:
        # Send surplus to planet closest to the action (most enemies nearby)
        def enemy_pressure(p):
            return sum(1 / max(1, dist(p.x, p.y, e.x, e.y)) for e in enemy_planets)

        staging = max(my_planets, key=enemy_pressure) if enemy_planets else max(my_planets, key=lambda p: p.production)

        for src in my_planets:
            if src.id == staging.id:
                continue
            rank    = prod_rank.get(src.id, 0.0)
            floor   = garrison_floor(src, planet_threat.get(src.id, 0), rank)
            surplus = (src.ships - committed[src.id]) - floor
            if surplus >= SURPLUS_THRESHOLD:
                angle = math.atan2(staging.y - src.y, staging.x - src.x)
                if not path_crosses_sun(src.x, src.y, staging.x, staging.y):
                    moves.append([src.id, angle, surplus])
                    committed[src.id] += surplus

    return moves