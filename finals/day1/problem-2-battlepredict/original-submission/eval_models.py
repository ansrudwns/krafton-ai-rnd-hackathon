import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.optimize import minimize

def parse_duels(events_str):
    duels = []
    for event in str(events_str).split():
        if '>' in event:
            winner, loser = event.split('>')
            duels.append((int(winner), int(loser)))
    return duels

df = pd.read_csv('[KRAFTON AI R&D HACKATHON] DAY1 P2 dataset.csv')
df_train = df[df['day'] != '22'].copy()
df_train = df_train[df_train['events'].str.contains('>')]

duels_by_day = defaultdict(list)
for _, row in df_train.iterrows():
    day = str(row['day'])
    # group by day. '?' are just treated as mid-days.
    parsed = parse_duels(row['events'])
    if day == '21':
        duels_by_day['val'].extend(parsed)
    else:
        duels_by_day['train'].extend(parsed)

train_duels = duels_by_day['train']
val_duels = duels_by_day['val']

# 1. Empirical Matrix with Laplace Smoothing
alpha = 1.0 # Laplace smoothing
wins = np.zeros((10, 10))
matches = np.zeros((10, 10))

for w, l in train_duels:
    wins[w, l] += 1
    matches[w, l] += 1
    matches[l, w] += 1

P_empirical = (wins + alpha) / (matches + 2*alpha)
# fallback for self matches or no matches
for i in range(10): P_empirical[i, i] = 0.5 

# 2. Bradley-Terry Model
# Minimize negative log likelihood
def neg_log_likelihood(skills, duels):
    loss = 0
    for w, l in duels:
        # P(w > l) = 1 / (1 + exp(-(s_w - s_l)))
        diff = skills[w] - skills[l]
        # loss += -log(1 / (1 + exp(-diff))) = log(1 + exp(-diff))
        loss += np.log1p(np.exp(-diff))
    return loss + 0.01 * np.sum(skills**2) # small L2 regularizer

res = minimize(neg_log_likelihood, np.zeros(10), args=(train_duels,), method='L-BFGS-B')
bt_skills = res.x

def get_bt_prob(w, l, skills):
    return 1.0 / (1.0 + np.exp(-(skills[w] - skills[l])))

P_bt = np.zeros((10, 10))
for i in range(10):
    for j in range(10):
        if i != j:
            P_bt[i, j] = get_bt_prob(i, j, bt_skills)

# Evaluation on Val (Day 21)
def log_loss(duels, P):
    loss = 0
    for w, l in duels:
        prob = P[w, l]
        loss -= np.log(prob)
    return loss / len(duels)

def acc(duels, P):
    correct = sum(1 for w, l in duels if P[w, l] > 0.5)
    return correct / len(duels)

ll_emp = log_loss(val_duels, P_empirical)
acc_emp = acc(val_duels, P_empirical)

ll_bt = log_loss(val_duels, P_bt)
acc_bt = acc(val_duels, P_bt)

print("Validation on Day 21 (Hold-out):")
print(f"Empirical Matrix Smoothing - LogLoss: {ll_emp:.4f}, Accuracy: {acc_emp:.4f}")
print(f"Bradley-Terry Model        - LogLoss: {ll_bt:.4f}, Accuracy: {acc_bt:.4f}")
print(f"Total train duels: {len(train_duels)}")
print(f"Total val duels: {len(val_duels)}")
