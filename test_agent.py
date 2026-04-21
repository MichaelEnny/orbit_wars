"""
Orbit Wars Agent Test Suite
Tests: (1) win vs random, (2) turn-by-turn stats, (3) speed, (4) logic checks
"""

import math
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# ── helpers ──────────────────────────────────────────────────────────────────

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"
INFO = "\033[94m INFO\033[0m"
WARN = "\033[93m WARN\033[0m"
SEP  = "-" * 60


def header(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")


def result(label, ok, detail=""):
    mark = PASS if ok else FAIL
    print(f"[{mark}] {label}" + (f" — {detail}" if detail else ""))


# ── import agent internals ────────────────────────────────────────────────────

from main import (
    fleet_speed, travel_time, dist,
    path_crosses_sun, agent,
    SUN_X, SUN_Y, SUN_RADIUS,
    planet_pos, aim_at_moving_target,
    enemy_incoming, friendly_incoming_count,
    score_target_fn,
)

# adapt test shims
def planet_position(p, turns, av, ip_dict):
    ip = ip_dict.get(p.id, p)
    return planet_pos(p.x, p.y, p.radius, turns, av, ip.x, ip.y)

def incoming_enemy_ships(planet, fleets, player):
    return enemy_incoming(planet, fleets, player)

def target_score(src, tgt, av, ip, comet_ids, fer, step):
    return score_target_fn(src, tgt, av, ip, comet_ids, fer, step)
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4 — Logic checks (run first, cheapest)
# ══════════════════════════════════════════════════════════════════════════════

header("TEST 4 — Logic Checks")

# fleet_speed
s1 = fleet_speed(1)
result("1 ship → speed = 1.0", abs(s1 - 1.0) < 1e-9, f"got {s1:.4f}")

s1000 = fleet_speed(1000)
result("1000 ships → speed ≈ 6.0", abs(s1000 - 6.0) < 0.01, f"got {s1000:.4f}")

s500 = fleet_speed(500)
result("500 ships → speed in [4.5, 5.5]", 4.5 < s500 < 5.5, f"got {s500:.4f}")

result("Larger fleet always faster", fleet_speed(100) < fleet_speed(500) < fleet_speed(999))

# dist
result("dist(0,0,3,4) = 5.0", abs(dist(0, 0, 3, 4) - 5.0) < 1e-9)
result("dist same point = 0", dist(10, 10, 10, 10) == 0.0)

# travel_time
tt = travel_time(0, 0, 6, 0, 1)  # 1 ship, speed=1 → time=6
result("travel_time 1-ship over dist=6 → 6 turns", abs(tt - 6.0) < 1e-6, f"got {tt:.4f}")

# path_crosses_sun
# Direct path through center
crosses = path_crosses_sun(0, 50, 100, 50)   # horizontal through (50,50)
result("Horizontal path through sun → crosses", crosses)

# Path that misses the sun
misses = path_crosses_sun(0, 0, 20, 0)        # far from center
result("Short path in corner → no cross", not misses)

# Tangent path along edge of sun
near = path_crosses_sun(0, 50 - SUN_RADIUS + 0.5, 100, 50 - SUN_RADIUS + 0.5)
result("Path tangent to sun edge → crosses", near)

# orbit prediction — static planet should not move
static_planet = Planet(0, -1, 95.0, 95.0, 1.0, 20, 2)  # orbital_radius ~63.6 > ROTATION_RADIUS_LIMIT
ip = {0: static_planet}
px, py = planet_position(static_planet, 50, 0.03, ip)
result("Static planet doesn't move", abs(px - 95.0) < 1e-6 and abs(py - 95.0) < 1e-6,
       f"got ({px:.2f},{py:.2f})")

# orbit prediction — orbiting planet should move
orb_r = 20.0  # well within ROTATION_RADIUS_LIMIT (< 50 - planet_radius)
orb_planet = Planet(1, -1, SUN_X + orb_r, SUN_Y, 1.0, 10, 1)
ip2 = {1: orb_planet}
omega = 0.03
px2, py2 = planet_position(orb_planet, 100, omega, ip2)
expected_angle = 0.0 + omega * 100   # initial angle=0, advanced 100 turns
ex = SUN_X + orb_r * math.cos(expected_angle)
ey = SUN_Y + orb_r * math.sin(expected_angle)
err = dist(px2, py2, ex, ey)
result("Orbiting planet moves correctly after 100 turns", err < 0.5,
       f"error={err:.4f}, got ({px2:.2f},{py2:.2f}) expected ({ex:.2f},{ey:.2f})")

# aim_at_moving_target — angle points toward target
src = Planet(0, 0, 10.0, 50.0, 1.5, 50, 3)
tgt = Planet(1, -1, 80.0, 50.0, 1.0, 5, 1)  # static, same y → angle should be ~0
angle, eta, tx, ty = aim_at_moving_target(src.x, src.y, 50,
    tgt.x, tgt.y, tgt.radius, 0.0, tgt.x, tgt.y)
result("Aim at static target on same row → angle ≈ 0", abs(angle) < 0.05,
       f"angle={angle:.4f} rad")

# target_score — high production beats low production at same distance
hi_prod = Planet(10, -1, 30.0, 50.0, 2.6, 5, 5)
lo_prod = Planet(11, -1, 30.0, 50.0, 1.0, 5, 1)
src2 = Planet(0, 0, 10.0, 50.0, 1.5, 100, 3)
score_hi = target_score(src2, hi_prod, 0.0, {}, set(), 0, 0)
score_lo = target_score(src2, lo_prod, 0.0, {}, set(), 0, 0)
result("High-production planet scores higher than low-production", score_hi > score_lo,
       f"hi={score_hi:.3f} lo={score_lo:.3f}")

# incoming_enemy_ships — fleet aimed at planet detected
planet_under_attack = Planet(5, 0, 50.0, 30.0, 1.5, 20, 2)
angle_to_planet = math.atan2(30.0 - 10.0, 50.0 - 10.0)
enemy_fleet = Fleet(0, 1, 10.0, 10.0, angle_to_planet, 99, 40)
detected = incoming_enemy_ships(planet_under_attack, [enemy_fleet], player=0)
result("Enemy fleet aimed at our planet is detected", detected == 40, f"detected={detected}")

# friendly fleet NOT counted as threat
friendly_fleet = Fleet(1, 0, 10.0, 10.0, angle_to_planet, 99, 30)
friendly_detected = incoming_enemy_ships(planet_under_attack, [friendly_fleet], player=0)
result("Friendly fleet not counted as enemy", friendly_detected == 0, f"got={friendly_detected}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 — Win vs random (3 games)
# ══════════════════════════════════════════════════════════════════════════════

header("TEST 1 — Win vs Random Bot (3 games)")

try:
    from kaggle_environments import make

    wins = 0
    losses = 0
    draws = 0
    agent_file = os.path.join(os.path.dirname(__file__), "main.py")

    for game_num in range(1, 4):
        env = make("orbit_wars", debug=False)
        env.run([agent_file, "random"])
        final = env.steps[-1]
        steps_played = len(env.steps)
        our_reward  = final[0].reward
        their_reward = final[1].reward

        if our_reward is None or their_reward is None:
            status = "ERROR"
            draws += 1
        elif our_reward > their_reward:
            status = "WIN"
            wins += 1
        elif our_reward < their_reward:
            status = "LOSS"
            losses += 1
        else:
            status = "DRAW"
            draws += 1

        our_status   = final[0].status
        their_status = final[1].status
        print(f"  Game {game_num}: {status} | turns={steps_played} "
              f"| us={our_reward} ({our_status}) "
              f"| random={their_reward} ({their_status})")

    total = wins + losses + draws
    result(f"Win rate vs random: {wins}/{total}",
           wins > losses,
           f"{wins}W {losses}L {draws}D")

except Exception as e:
    print(f"[{FAIL}] Could not run game: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 — Turn-by-turn stats (1 full game)
# ══════════════════════════════════════════════════════════════════════════════

header("TEST 2 — Turn-by-Turn Stats (1 full game)")

try:
    from kaggle_environments import make

    env = make("orbit_wars", debug=False)
    agent_file = os.path.join(os.path.dirname(__file__), "main.py")
    env.run([agent_file, "random"])

    print(f"  {'Turn':>4}  {'OurPlanets':>10}  {'OurShips':>8}  {'EnemyPlanets':>12}  {'EnemyShips':>10}  {'Fleets':>6}")
    print(f"  {'----':>4}  {'----------':>10}  {'--------':>8}  {'------------':>12}  {'----------':>10}  {'------':>6}")

    sample_turns = list(range(0, len(env.steps), max(1, len(env.steps) // 20)))
    if len(env.steps) - 1 not in sample_turns:
        sample_turns.append(len(env.steps) - 1)

    for t in sample_turns:
        step = env.steps[t]
        obs = step[0].observation
        if obs is None:
            continue
        planets = [Planet(*p) for p in obs.get("planets", [])]
        fleets  = obs.get("fleets", [])
        player  = 0

        our_planets   = [p for p in planets if p.owner == player]
        enemy_planets = [p for p in planets if p.owner not in (-1, player)]
        our_ships     = sum(p.ships for p in our_planets) + sum(f[6] for f in fleets if f[1] == player)
        enemy_ships   = sum(p.ships for p in enemy_planets) + sum(f[6] for f in fleets if f[1] not in (-1, player))

        print(f"  {t:>4}  {len(our_planets):>10}  {our_ships:>8}  {len(enemy_planets):>12}  {enemy_ships:>10}  {len(fleets):>6}")

    final = env.steps[-1]
    our_r = final[0].reward
    their_r = final[1].reward
    print(f"\n  Final score — us: {our_r}  |  random: {their_r}")
    outcome = "WIN" if our_r > their_r else ("LOSS" if our_r < their_r else "DRAW")
    result(f"Game completed ({outcome})", True, f"{len(env.steps)} turns played")

except Exception as e:
    print(f"[{FAIL}] Stats game failed: {e}")
    import traceback; traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 — Speed test (must stay under 1s/turn)
# ══════════════════════════════════════════════════════════════════════════════

header("TEST 3 — Speed Test (1s budget per turn)")

try:
    from kaggle_environments import make

    env = make("orbit_wars", debug=False)
    agent_file = os.path.join(os.path.dirname(__file__), "main.py")

    # Collect observations from a real game first
    env.run([agent_file, "random"])

    times = []
    slow_turns = []

    for t, step in enumerate(env.steps):
        obs = step[0].observation
        if obs is None:
            continue
        obs_dict = dict(obs)
        t0 = time.perf_counter()
        try:
            moves = agent(obs_dict)
        except Exception as e:
            print(f"  Turn {t}: agent crashed — {e}")
            continue
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        if elapsed > 1.0:
            slow_turns.append((t, elapsed))

    if times:
        avg_ms  = sum(times) / len(times) * 1000
        max_ms  = max(times) * 1000
        p99_ms  = sorted(times)[int(len(times) * 0.99)] * 1000
        over_budget = len(slow_turns)

        print(f"  Turns measured : {len(times)}")
        print(f"  Avg time       : {avg_ms:.2f} ms")
        print(f"  Max time       : {max_ms:.2f} ms")
        print(f"  p99 time       : {p99_ms:.2f} ms")
        print(f"  Turns > 1000ms : {over_budget}")
        if slow_turns:
            for tt, te in slow_turns[:5]:
                print(f"    Turn {tt}: {te*1000:.1f} ms")

        result("Average turn < 100ms",  avg_ms < 100,  f"{avg_ms:.2f} ms")
        result("Max turn < 1000ms",     max_ms < 1000, f"{max_ms:.2f} ms")
        result("No turns exceeded budget", over_budget == 0, f"{over_budget} slow turns")

except Exception as e:
    print(f"[{FAIL}] Speed test failed: {e}")
    import traceback; traceback.print_exc()


print(f"\n{SEP}\n  All tests complete.\n{SEP}\n")
