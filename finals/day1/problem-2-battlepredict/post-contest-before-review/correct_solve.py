"""
BattlePredict - Correct Solution
=================================
Intended pipeline:
  1. Parse dataset
  2. Fit BTL on each labeled day (1, 11, 21) -> observe non-stationarity
  3. Fit BTL on each anonymous block (18 blocks)
  4. Recover hidden day labels via Hungarian assignment
  5. Fit linear skill trend s_i(d) = w_i*d + b_i for each player
  6. Simulate each test gauntlet using day-specific skills -> expected kills
"""

import csv
import math
from itertools import permutations
import numpy as np

# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def fit_btl(duels, epochs=1000, lr=0.05, l2=0.05):
    """Fit Bradley-Terry model. Returns mean-centered skill vector."""
    if len(duels) < 5:
        return [0.0] * 10
    skills = [0.0] * 10
    for _ in range(epochs):
        grads = [0.0] * 10
        for w, l in duels:
            p = sigmoid(skills[w] - skills[l])
            grads[w] += (1.0 - p)
            grads[l] -= (1.0 - p)
        for i in range(10):
            skills[i] += lr * (grads[i] - l2 * skills[i])
    mean = sum(skills) / 10
    return [s - mean for s in skills]


def hungarian_small(cost):
    """
    Exact Hungarian via brute-force permutation (fine for 9x9).
    cost: n x m list-of-lists (n <= m).
    Returns (row_indices, col_indices).
    """
    n = len(cost)
    m = len(cost[0])
    best_cost = float('inf')
    best_perm = None
    for perm in permutations(range(m), n):
        c = sum(cost[i][perm[i]] for i in range(n))
        if c < best_cost:
            best_cost = c
            best_perm = perm
    return list(range(n)), list(best_perm)


# ─────────────────────────────────────────────
# Step 1: Parse dataset
# ─────────────────────────────────────────────

CSV_FILE = '[KRAFTON AI R&D HACKATHON] DAY1 P2 dataset.csv'

labeled_duels   = {}   # day (int) -> list of (winner, loser)
anon_blocks     = []   # list of lists of (winner, loser)  [18 blocks]
test_day_gauntlets = {}  # day (int) -> list of gauntlets

current_block = None

with open(CSV_FILE, encoding='utf-8') as f:
    for row in csv.DictReader(f):
        day      = row['day'].strip()
        mid      = int(row['match_in_day'])
        events   = row['events'].strip().split()

        # --- Labeled training days ---
        if day in ('1', '11', '21'):
            d = int(day)
            labeled_duels.setdefault(d, [])
            for e in events:
                if '>' in e:
                    w, l = map(int, e.split('>'))
                    labeled_duels[d].append((w, l))

        # --- Anonymous blocks ---
        elif day == '?':
            if mid == 1:                          # new block starts
                current_block = []
                anon_blocks.append(current_block)
            if current_block is not None:
                for e in events:
                    if '>' in e:
                        w, l = map(int, e.split('>'))
                        current_block.append((w, l))

        # --- Test days (22-50): gauntlet orders only ---
        else:
            d = int(day)
            if d >= 22 and not any('>' in e for e in events):
                test_day_gauntlets.setdefault(d, []).append(
                    [int(x) for x in events]
                )

print(f"Labeled days     : {sorted(labeled_duels.keys())}")
print(f"Anonymous blocks : {len(anon_blocks)}")
total_test = sum(len(g) for g in test_day_gauntlets.values())
print(f"Test gauntlets   : {total_test}  (days {min(test_day_gauntlets)}-{max(test_day_gauntlets)})")


# ─────────────────────────────────────────────
# Step 2: BTL on labeled days & anonymous blocks
# ─────────────────────────────────────────────

labeled_skills = {}
for d, duels in labeled_duels.items():
    labeled_skills[d] = np.array(fit_btl(duels))

print("\nBTL skills on labeled days:")
print("Day  | " + "  ".join(f"P{i}" for i in range(10)))
for d in sorted(labeled_skills):
    sk = labeled_skills[d]
    print(f"  {d:2d} | " + "  ".join(f"{s:+.2f}" for s in sk))

anon_skills = []
for i, duels in enumerate(anon_blocks):
    anon_skills.append(np.array(fit_btl(duels)))


# ─────────────────────────────────────────────
# Step 3: Recover hidden day labels via Hungarian
# ─────────────────────────────────────────────
# Linear anchor: s_pred(d) = s(1) + (d-1) * slope
#                slope = (s(21) - s(1)) / 20

slope_anchor = (labeled_skills[21] - labeled_skills[1]) / 20.0

def predicted_skill_at(d):
    return labeled_skills[1] + (d - 1) * slope_anchor


def cost_matrix(block_range, day_range):
    """L2 distance between each block's BTL skills and predicted skills for each day."""
    C = []
    for bi in block_range:
        row = []
        for d in day_range:
            diff = anon_skills[bi] - predicted_skill_at(d)
            row.append(float(np.sum(diff ** 2)))
        C.append(row)
    return C


# Group 1: first 9 blocks -> Days 2-10
C1 = cost_matrix(range(9), range(2, 11))
r1, c1 = hungarian_small(C1)

# Group 2: next 9 blocks -> Days 12-20
C2 = cost_matrix(range(9, 18), range(12, 21))
r2, c2 = hungarian_small(C2)

recovered_day = {}
print("\nRecovered day labels:")
for r, c in zip(r1, c1):
    day = list(range(2, 11))[c]
    recovered_day[r] = day
    print(f"  Block {r:2d} -> Day {day}  (cost={C1[r][c]:.2f})")
for r, c in zip(r2, c2):
    day = list(range(12, 21))[c]
    recovered_day[9 + r] = day
    print(f"  Block {9+r:2d} -> Day {day}  (cost={C2[r][c]:.2f})")


# ─────────────────────────────────────────────
# Step 4: Fit linear skill trend
#   s_i(d) = w_i * d + b_i  via least squares
# ─────────────────────────────────────────────

# Collect all (day, skills) pairs
all_day_skills = {}
for d, sk in labeled_skills.items():
    all_day_skills[d] = sk
for block_idx, day in recovered_day.items():
    all_day_skills[day] = anon_skills[block_idx]

# Re-fit BTL on raw duels for recovered days (more accurate than using anon_skills directly)
all_day_duels = {}
for d, duels in labeled_duels.items():
    all_day_duels[d] = duels
for block_idx, day in recovered_day.items():
    all_day_duels[day] = anon_blocks[block_idx]

refined_skills = {}
for d, duels in all_day_duels.items():
    refined_skills[d] = np.array(fit_btl(duels, epochs=1000))

# Least-squares linear fit per player
days_sorted = sorted(refined_skills.keys())
days_np     = np.array(days_sorted, dtype=float)
A           = np.column_stack([days_np, np.ones_like(days_np)])  # design matrix

linear_params = []  # list of (w_i, b_i)
print("\nLinear skill trend per player:")
print("Player | w (slope/day) | b (intercept)")
for p in range(10):
    y        = np.array([refined_skills[d][p] for d in days_sorted])
    (w, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    linear_params.append((w, b))
    direction = "↑" if w > 0.01 else ("↓" if w < -0.01 else "→")
    print(f"  P{p}   | {w:+.4f}         | {b:+.4f}  {direction}")


# ─────────────────────────────────────────────
# Step 5: Simulate test gauntlets
#   For each day d in 22-50, compute skills from linear model,
#   then analytically propagate kill probabilities through the gauntlet.
# ─────────────────────────────────────────────

def skill_at(p, d):
    w, b = linear_params[p]
    return w * d + b


def simulate_gauntlet(gauntlet, day):
    """Returns expected kill count per player for one gauntlet."""
    kills = [0.0] * 10
    p1, p2 = gauntlet[0], gauntlet[1]

    p_win_1 = sigmoid(skill_at(p1, day) - skill_at(p2, day))
    survivors = {p1: p_win_1, p2: 1.0 - p_win_1}
    kills[p1] += p_win_1
    kills[p2] += 1.0 - p_win_1

    for challenger in gauntlet[2:]:
        next_survivors = {challenger: 0.0}
        for surv, p_surv in survivors.items():
            p_surv_wins = sigmoid(skill_at(surv, day) - skill_at(challenger, day))
            p_chal_wins = 1.0 - p_surv_wins
            kills[surv]     += p_surv * p_surv_wins
            kills[challenger] += p_surv * p_chal_wins
            next_survivors[surv]       = next_survivors.get(surv, 0.0) + p_surv * p_surv_wins
            next_survivors[challenger] += p_surv * p_chal_wins
        survivors = next_survivors

    return kills


total_kills = [0.0] * 10

for d in sorted(test_day_gauntlets.keys()):
    for gauntlet in test_day_gauntlets[d]:
        k = simulate_gauntlet(gauntlet, d)
        for p in range(10):
            total_kills[p] += k[p]


# ─────────────────────────────────────────────
# Step 6: Results
# ─────────────────────────────────────────────

original_submission = [1019.08, 908.34, 798.16, 687.05, 449.52,
                       547.12,  419.54, 302.23, 418.85, 250.09]

print("\n" + "="*60)
print("FINAL RESULTS — Correct vs Original Submission")
print("="*60)
print(f"{'Player':<8} {'Correct':>10} {'Original':>10} {'Diff':>10}")
print("-"*40)
for p in range(10):
    diff = total_kills[p] - original_submission[p]
    print(f"  P{p}   {total_kills[p]:>10.1f} {original_submission[p]:>10.1f} {diff:>+10.1f}")
print("-"*40)
print(f"  Total {sum(total_kills):>10.1f} {sum(original_submission):>10.1f}")

print("\n=== Submission Format (user_0 .. user_9) ===")
for p, k in enumerate(total_kills):
    print(f"user_{p}: {k:.2f}")

checksum = sum(total_kills)
print(f"\nChecksum: {checksum:.4f}  (should be 5800)")
