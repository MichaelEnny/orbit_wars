"""
Orbit Wars v5 — Forward-Planning Agent

Architecture:
  1. Fast game-state simulator (approximate physics, no collision detection overhead)
  2. Candidate move generator (from v4 heuristics)
  3. Greedy beam search: simulate N turns ahead for each candidate set, pick best
  4. Board evaluator: expected ship-count delta, production advantage, territory control
  5. Fall back to best heuristic move if time budget is tight
"""

import math
import time
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet, CENTER, ROTATION_RADIUS_LIMIT

# ---------- physics constants ----------
MAX_SPEED    = 6.0
MIN_SPEED    = 1.0
SUN_X, SUN_Y = 50.0, 50.0
SUN_RADIUS   = 10.0
BOARD        = 100.0

# ---------- tuning ----------
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
INTERCEPT_WINDOW     = 10
INTERCEPT_MIN_SHIPS  = 30

# lookahead
LOOKAHEAD_TURNS      = 6    # simulate this many turns ahead per candidate
LOOKAHEAD_CANDIDATES = 6    # how many candidate move-sets to evaluate
TIME_BUDGET          = 0.70 # seconds — fall back if we exceed this


# ═══════════════════════════════════════════════════════════
# PHYSICS HELPERS
# ═══════════════════════════════════════════════════════════

def fleet_speed(ships):
    if ships <= 1:
        return MIN_SPEED
    return MIN_SPEED + (MAX_SPEED - MIN_SPEED) * (math.log(ships) / math.log(1000)) ** 1.5


def dist(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def travel_time(x1, y1, x2, y2, ships):
    d = dist(x1, y1, x2, y2)
    return 0.0 if d < 1e-9 else d / fleet_speed(max(1, ships))


def planet_pos(px, py, pradius, turns, angular_vel, init_x, init_y):
    """Return future (x,y) of a planet given its initial position."""
    orbital_r = dist(px, py, SUN_X, SUN_Y)
    if orbital_r + pradius < ROTATION_RADIUS_LIMIT:
        r = dist(init_x, init_y, SUN_X, SUN_Y)
        cur_angle = math.atan2(py - SUN_Y, px - SUN_X)
        fut_angle = cur_angle + angular_vel * turns
        return SUN_X + r * math.cos(fut_angle), SUN_Y + r * math.sin(fut_angle)
    return px, py


def aim_at_moving_target(src_x, src_y, ships, target_x, target_y,
                          target_r, angular_vel, init_x, init_y):
    tx, ty = target_x, target_y
    for _ in range(MAX_LEAD_ITERS):
        eta = travel_time(src_x, src_y, tx, ty, ships)
        tx, ty = planet_pos(target_x, target_y, target_r, eta, angular_vel, init_x, init_y)
    return math.atan2(ty - src_y, tx - src_x), eta, tx, ty


def path_crosses_sun(x1, y1, x2, y2):
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


def angle_diff(a, b):
    return abs((a - b + math.pi) % (2*math.pi) - math.pi)


# ═══════════════════════════════════════════════════════════
# FAST GAME STATE  (lightweight dicts, no named tuples)
# ═══════════════════════════════════════════════════════════
# Planet dict keys: id, owner, x, y, radius, ships, production, init_x, init_y
# Fleet  dict keys: id, owner, x, y, angle, ships, speed, from_pid

def make_sim_state(planets, fleets, angular_vel):
    """Convert Planet/Fleet named tuples into mutable sim dicts."""
    p_map = {}
    for p in planets:
        p_map[p.id] = {
            "id": p.id, "owner": p.owner,
            "x": p.x, "y": p.y,
            "radius": p.radius, "ships": float(p.ships),
            "production": p.production,
            "init_x": p.x, "init_y": p.y,  # updated by caller
        }
    f_list = []
    for f in fleets:
        f_list.append({
            "id": f.id, "owner": f.owner,
            "x": f.x, "y": f.y,
            "angle": f.angle, "ships": float(f.ships),
            "speed": fleet_speed(f.ships),
        })
    return {"planets": p_map, "fleets": f_list, "angular_vel": angular_vel, "next_fid": 100000}


def clone_state(state):
    return {
        "planets": {k: dict(v) for k, v in state["planets"].items()},
        "fleets":  [dict(f) for f in state["fleets"]],
        "angular_vel": state["angular_vel"],
        "next_fid": state["next_fid"],
    }


def apply_moves_to_state(state, moves):
    """Launch fleets from moves = [[pid, angle, ships], ...]"""
    for pid, angle, ships in moves:
        p = state["planets"].get(pid)
        if p is None:
            continue
        ships = int(min(ships, p["ships"]))
        if ships < 1:
            continue
        p["ships"] -= ships
        speed = fleet_speed(ships)
        state["fleets"].append({
            "id": state["next_fid"], "owner": p["owner"],
            "x": p["x"], "y": p["y"],
            "angle": angle, "ships": float(ships), "speed": speed,
        })
        state["next_fid"] += 1


def sim_step(state):
    """Advance state by one turn (approximate — no comet movement, simplified combat)."""
    p_map = state["planets"]
    angular_vel = state["angular_vel"]

    # Move fleets
    to_remove = []
    combat = {}  # planet_id -> {owner: ships}
    for f in state["fleets"]:
        nx = f["x"] + math.cos(f["angle"]) * f["speed"]
        ny = f["y"] + math.sin(f["angle"]) * f["speed"]

        # Destroy if OOB or crosses sun
        if not (0 <= nx <= BOARD and 0 <= ny <= BOARD):
            to_remove.append(f["id"])
            continue
        if path_crosses_sun(f["x"], f["y"], nx, ny):
            to_remove.append(f["id"])
            continue

        # Check planet collision
        hit = None
        for p in p_map.values():
            if dist(nx, ny, p["x"], p["y"]) <= p["radius"] + f["speed"]:
                hit = p
                break
        if hit:
            pid = hit["id"]
            if pid not in combat:
                combat[pid] = {}
            o = f["owner"]
            combat[pid][o] = combat[pid].get(o, 0) + f["ships"]
            to_remove.append(f["id"])
        else:
            f["x"], f["y"] = nx, ny

    state["fleets"] = [f for f in state["fleets"] if f["id"] not in set(to_remove)]

    # Resolve combat
    for pid, attackers in combat.items():
        p = p_map[pid]
        # Add garrison as defender
        if p["owner"] >= 0:
            attackers[p["owner"]] = attackers.get(p["owner"], 0) + p["ships"]

        if not attackers:
            continue

        sorted_forces = sorted(attackers.items(), key=lambda x: -x[1])
        if len(sorted_forces) == 1:
            winner, ships = sorted_forces[0]
        else:
            top_owner, top_ships = sorted_forces[0]
            second_ships = sorted_forces[1][1]
            survivor = top_ships - second_ships
            if survivor <= 0:
                # Tie — planet stays with current owner, garrison zeroed
                p["ships"] = 0
                continue
            winner, ships = top_owner, survivor
            # Winner fights garrison if different owner
            if winner != p["owner"] and p["owner"] >= 0:
                garrison_contrib = attackers.get(p["owner"], 0)
                survivor2 = ships - garrison_contrib
                if survivor2 > 0:
                    p["owner"] = winner
                    p["ships"] = survivor2
                else:
                    p["ships"] = max(0, garrison_contrib - ships)
                continue

        if winner == p["owner"]:
            p["ships"] = ships
        else:
            p["owner"] = winner
            p["ships"] = ships

    # Production
    for p in p_map.values():
        if p["owner"] >= 0:
            p["ships"] += p["production"]

    # Rotate orbiting planets
    for p in p_map.values():
        orbital_r = dist(p["x"], p["y"], SUN_X, SUN_Y)
        if orbital_r + p["radius"] < ROTATION_RADIUS_LIMIT:
            cur_angle = math.atan2(p["y"] - SUN_Y, p["x"] - SUN_X)
            new_angle = cur_angle + angular_vel
            r = orbital_r
            p["x"] = SUN_X + r * math.cos(new_angle)
            p["y"] = SUN_Y + r * math.sin(new_angle)


# ═══════════════════════════════════════════════════════════
# BOARD EVALUATOR
# ═══════════════════════════════════════════════════════════

def evaluate_state(state, player):
    """
    Score the board state from `player`'s perspective.
    Higher = better for player.
    """
    my_ships      = 0.0
    enemy_ships   = 0.0
    my_prod       = 0.0
    enemy_prod    = 0.0
    my_planets    = 0
    enemy_planets = 0

    for p in state["planets"].values():
        if p["owner"] == player:
            my_ships   += p["ships"]
            my_prod    += p["production"]
            my_planets += 1
        elif p["owner"] >= 0:
            enemy_ships   += p["ships"]
            enemy_prod    += p["production"]
            enemy_planets += 1

    for f in state["fleets"]:
        if f["owner"] == player:
            my_ships += f["ships"]
        elif f["owner"] >= 0:
            enemy_ships += f["ships"]

    # Production advantage is compounding — weight it heavily
    prod_advantage = (my_prod - enemy_prod) * 15.0
    ship_advantage = my_ships - enemy_ships
    territory      = (my_planets - enemy_planets) * 5.0

    # Winning bonus: if enemy has no planets/fleets
    if enemy_planets == 0 and enemy_ships == 0:
        return 1e9
    if my_planets == 0 and my_ships == 0:
        return -1e9

    return ship_advantage + prod_advantage + territory


def simulate_and_score(state, moves, player, turns):
    """Apply moves, simulate `turns` steps, return evaluation score."""
    s = clone_state(state)
    apply_moves_to_state(s, moves)
    for _ in range(turns):
        sim_step(s)
    return evaluate_state(s, player)


# ═══════════════════════════════════════════════════════════
# FLEET / THREAT HELPERS  (same as v4)
# ═══════════════════════════════════════════════════════════

def fleets_aimed_at(planet_x, planet_y, fleets, owner_filter):
    result = []
    for f in fleets:
        if not owner_filter(f.owner):
            continue
        expected = math.atan2(planet_y - f.y, planet_x - f.x)
        if angle_diff(f.angle, expected) < FLEET_ANGLE_TOL:
            result.append(f)
    return result


def enemy_incoming(planet, fleets, player):
    fs = fleets_aimed_at(planet.x, planet.y, fleets, lambda o: o not in (-1, player))
    return sum(f.ships for f in fs)


def friendly_incoming_count(planet, fleets, player):
    fs = fleets_aimed_at(planet.x, planet.y, fleets, lambda o: o == player)
    return sum(f.ships for f in fs)


def comet_turns_left(comet_id, planet_id_to_comet):
    group = planet_id_to_comet.get(comet_id)
    if group is None:
        return 999
    path = group.get("paths", [[]])[0]
    idx  = group.get("path_index", 0)
    return max(0, len(path) - idx)


def garrison_floor(planet, enemy_threat, prod_rank):
    base = max(MIN_GARRISON, int(planet.ships * (GARRISON_FRAC + 0.10 * prod_rank)))
    if enemy_threat > 0:
        base = max(base, int(enemy_threat * 1.1) + ATTACK_BUFFER)
    return base


# ═══════════════════════════════════════════════════════════
# CANDIDATE MOVE GENERATOR  (v4 heuristics → list of move-sets)
# ═══════════════════════════════════════════════════════════

def score_target_fn(src, target, angular_vel, initial_planets, comet_ids,
                    f_en_route, step, sendable=1):
    ip = initial_planets.get(target.id, target)
    tx, ty = planet_pos(target.x, target.y, target.radius, 0,
                        angular_vel, ip.x, ip.y)
    d = dist(src.x, src.y, tx, ty)
    if d < 1e-6:
        return -999.0

    prod = target.production * (COMET_BONUS if target.id in comet_ids else 1.0)
    cost = max(1, target.ships + ATTACK_BUFFER - f_en_route)
    score = (prod * PRODUCTION_WEIGHT) / (d + cost)

    if sendable >= MIN_MASS_FLEET:
        score *= MASS_FLEET_BONUS
    if path_crosses_sun(src.x, src.y, tx, ty):
        score *= 0.15
    if step < EARLY_RUSH_TURNS and target.owner == -1:
        score *= 1.5
    if step > 300 and target.owner == -1:
        score *= 0.6
    return score


def generate_candidates(my_planets, all_targets, fleets, angular_vel,
                         initial_planets, comet_ids, planet_id_to_comet,
                         planet_threat, friendly_en_route, prod_rank,
                         committed_base, step):
    """
    Generate LOOKAHEAD_CANDIDATES different move-sets to evaluate.
    Each candidate is a list of [pid, angle, ships] moves.
    Strategy variants:
      0: aggressive (attack hardest targets, low garrison)
      1: balanced   (default v4 logic)
      2: economic   (grab highest-production planets only)
      3: rush       (every planet attacks closest neutral)
      4: concentrate (all ships at single best target)
      5: defensive  (no attacks, just hold and grow)
    """
    candidates = []

    # Shared helper
    def build_moves(src_list, targets, garrison_mult, max_atk, rush=False, single_target=None):
        committed = dict(committed_base)
        moves = []
        for src in sorted(src_list, key=lambda p: -p.ships):
            threat = planet_threat.get(src.id, 0)
            rank   = prod_rank.get(src.id, 0.0)
            floor  = max(MIN_GARRISON, int(src.ships * GARRISON_FRAC * garrison_mult +
                                           0.10 * rank * src.ships))
            if threat > 0:
                floor = max(floor, int(threat * 1.1) + ATTACK_BUFFER)
            sendable = (src.ships - committed.get(src.id, 0)) - floor
            if sendable < 2:
                continue

            target_list = [single_target] if single_target else targets
            scored = []
            for t in target_list:
                fer = friendly_en_route.get(t.id, 0)
                s = score_target_fn(src, t, angular_vel, initial_planets,
                                    comet_ids, fer, step, sendable)
                scored.append((s, t))
            scored.sort(key=lambda x: -x[0])

            atk = 0
            rem = sendable
            for _, best in scored:
                if atk >= max_atk or rem < 2:
                    break
                ip = initial_planets.get(best.id, best)
                angle, eta, tx, ty = aim_at_moving_target(
                    src.x, src.y, rem,
                    best.x, best.y, best.radius,
                    angular_vel, ip.x, ip.y)
                if path_crosses_sun(src.x, src.y, tx, ty):
                    continue
                fer = friendly_en_route.get(best.id, 0)
                garr = best.ships if best.owner == -1 else best.ships + int(eta * best.production)
                needed = max(1, garr + ATTACK_BUFFER - fer)
                if needed > rem:
                    if rush and rem >= best.ships:
                        to_send = rem
                    elif best.owner != -1 and rem > best.ships * 0.5:
                        to_send = rem
                    else:
                        continue
                else:
                    to_send = needed
                to_send = min(to_send, rem)
                if to_send < 1:
                    continue
                moves.append([src.id, angle, to_send])
                committed[src.id] = committed.get(src.id, 0) + to_send
                rem  -= to_send
                atk  += 1
        return moves

    neutral_targets = [t for t in all_targets if t.owner == -1]
    enemy_targets   = [t for t in all_targets if t.owner not in (-1,)]
    high_prod       = sorted(all_targets, key=lambda t: -t.production)

    # 0: aggressive
    candidates.append(build_moves(my_planets, all_targets, 0.8, MAX_ATTACKS_PER_TURN, rush=True))

    # 1: balanced (default)
    candidates.append(build_moves(my_planets, all_targets, 1.0, MAX_ATTACKS_PER_TURN))

    # 2: economic — high-production targets only
    top_prod = high_prod[:max(3, len(high_prod)//2)]
    candidates.append(build_moves(my_planets, top_prod, 1.0, 2))

    # 3: rush closest neutral
    candidates.append(build_moves(my_planets, neutral_targets or all_targets,
                                   0.7, 1, rush=True))

    # 4: concentrate — best single target, all planets cooperate
    if all_targets:
        best_t = max(all_targets,
                     key=lambda t: t.production / max(1, t.ships) if my_planets
                     else 0)
        candidates.append(build_moves(my_planets, all_targets, 1.0, 1,
                                       single_target=best_t))
    else:
        candidates.append([])

    # 5: pure defense — no attacks
    candidates.append([])

    return candidates


# ═══════════════════════════════════════════════════════════
# DEFENSE  (same as v4)
# ═══════════════════════════════════════════════════════════

def build_defense_moves(my_planets, fleets, planet_threat, prod_rank, committed):
    moves = []
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
    return moves


# ═══════════════════════════════════════════════════════════
# SURPLUS REDISTRIBUTION
# ═══════════════════════════════════════════════════════════

def build_surplus_moves(my_planets, enemy_planets, planet_threat, prod_rank, committed):
    moves = []
    if not my_planets:
        return moves

    if enemy_planets:
        def pressure(p):
            return sum(1 / max(1, dist(p.x, p.y, e.x, e.y)) for e in enemy_planets)
        staging = max(my_planets, key=pressure)
    else:
        staging = max(my_planets, key=lambda p: p.production)

    for src in my_planets:
        if src.id == staging.id:
            continue
        rank  = prod_rank.get(src.id, 0.0)
        floor = garrison_floor(src, planet_threat.get(src.id, 0), rank)
        surplus = (src.ships - committed.get(src.id, 0)) - floor
        if surplus >= SURPLUS_THRESHOLD:
            angle = math.atan2(staging.y - src.y, staging.x - src.x)
            if not path_crosses_sun(src.x, src.y, staging.x, staging.y):
                moves.append([src.id, angle, surplus])
                committed[src.id] = committed.get(src.id, 0) + surplus
    return moves


# ═══════════════════════════════════════════════════════════
# MAIN AGENT
# ═══════════════════════════════════════════════════════════

def agent(obs):
    t_start = time.perf_counter()

    # ── parse ──────────────────────────────────────────────
    if isinstance(obs, dict):
        player      = obs.get("player", 0)
        raw_planets = obs.get("planets", [])
        raw_fleets  = obs.get("fleets", [])
        angular_vel = obs.get("angular_velocity", 0.0)
        raw_initial = obs.get("initial_planets", [])
        comet_ids   = set(obs.get("comet_planet_ids", []))
        raw_comets  = obs.get("comets", [])
        step        = obs.get("step", 0)
    else:
        player      = obs.player
        raw_planets = obs.planets
        raw_fleets  = obs.fleets
        angular_vel = obs.angular_velocity
        raw_initial = obs.initial_planets
        comet_ids   = set(obs.comet_planet_ids)
        raw_comets  = getattr(obs, "comets", [])
        step        = getattr(obs, "step", 0)

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

    # ── production rank ────────────────────────────────────
    if len(my_planets) > 1:
        min_p = min(p.production for p in my_planets)
        max_p = max(p.production for p in my_planets)
        rng   = max(1, max_p - min_p)
        prod_rank = {p.id: (p.production - min_p) / rng for p in my_planets}
    else:
        prod_rank = {my_planets[0].id: 1.0}

    # ── pre-compute threats ────────────────────────────────
    planet_threat     = {p.id: enemy_incoming(p, fleets, player) for p in my_planets}
    friendly_en_route = {p.id: friendly_incoming_count(p, fleets, player) for p in planets}
    committed         = {p.id: 0 for p in my_planets}

    # ── Phase 1: Defense (always runs, not part of lookahead) ─
    defense_moves = build_defense_moves(my_planets, fleets, planet_threat,
                                         prod_rank, committed)

    if not all_targets:
        surplus = build_surplus_moves(my_planets, enemy_planets, planet_threat,
                                       prod_rank, committed)
        return defense_moves + surplus

    # ── Phase 2: Build sim state ───────────────────────────
    sim_state = make_sim_state(planets, fleets, angular_vel)
    # Patch initial positions into sim planets
    for pid, ip in initial_planets.items():
        if pid in sim_state["planets"]:
            sim_state["planets"][pid]["init_x"] = ip.x
            sim_state["planets"][pid]["init_y"] = ip.y

    # Apply defense moves to committed tracker (already done above)
    # and to sim state so lookahead reflects them
    apply_moves_to_state(sim_state, defense_moves)

    # ── Phase 3: Generate candidates ──────────────────────
    candidates = generate_candidates(
        my_planets, all_targets, fleets, angular_vel, initial_planets,
        comet_ids, planet_id_to_comet, planet_threat, friendly_en_route,
        prod_rank, committed, step
    )

    # ── Phase 4: Lookahead evaluation ─────────────────────
    best_moves  = candidates[1]  # balanced as default
    best_score  = -1e18
    turns_to_sim = LOOKAHEAD_TURNS

    for i, cand_moves in enumerate(candidates):
        if time.perf_counter() - t_start > TIME_BUDGET:
            break  # time guard — use best found so far
        try:
            score = simulate_and_score(sim_state, cand_moves, player, turns_to_sim)
        except Exception:
            score = -1e18
        if score > best_score:
            best_score  = score
            best_moves  = cand_moves

    # ── Phase 5: Surplus redistribution ───────────────────
    # Update committed from chosen moves
    for pid, angle, ships in best_moves:
        committed[pid] = committed.get(pid, 0) + int(ships)

    surplus_moves = build_surplus_moves(my_planets, enemy_planets, planet_threat,
                                         prod_rank, committed)

    return defense_moves + best_moves + surplus_moves