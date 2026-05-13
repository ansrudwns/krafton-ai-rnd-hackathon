import csv
import math

def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        z = math.exp(x)
        return z / (1.0 + z)

def reconstruct_gauntlet(events):
    # events: list of "W>L"
    duels = []
    for e in events:
        w, l = e.split('>')
        duels.append((int(w), int(l)))
    
    # duel 1: p1 vs p2
    w1, l1 = duels[0]
    gauntlet = [w1, l1]
    last_winner = w1
    
    for i in range(1, 4):
        w, l = duels[i]
        # one of w, l must be last_winner
        if w == last_winner:
            new_player = l
        elif l == last_winner:
            new_player = w
        else:
            # fallback
            new_player = l
        gauntlet.append(new_player)
        last_winner = w
        
    return gauntlet

def main():
    csv_file = '[KRAFTON AI R&D HACKATHON] DAY1 P2 dataset.csv'
    
    train_duels = []
    day21_gauntlets = []
    day21_true_kills = [0.0]*10
    
    wins = [[0]*10 for _ in range(10)]
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            day = row['day']
            events_str = row['events']
            if not '>' in events_str:
                continue
                
            events = events_str.strip().split()
            
            if day == '21':
                # Hold-out set
                for e in events:
                    w, l = e.split('>')
                    day21_true_kills[int(w)] += 1.0
                gauntlet = reconstruct_gauntlet(events)
                day21_gauntlets.append(gauntlet)
            else:
                # Training set
                for e in events:
                    w, l = e.split('>')
                    w, l = int(w), int(l)
                    train_duels.append((w, l))
                    wins[w][l] += 1
                    
    # Fit BT
    num_players = 10
    skills = [0.0] * num_players
    epochs = 2000
    lr = 0.01
    lambda_val = 8.0
    N = len(train_duels)
    for epoch in range(epochs):
        grads = [0.0] * num_players
        for idx, (w, l) in enumerate(train_duels):
            weight = math.exp(-lambda_val * (N - 1 - idx) / N)
            p = sigmoid(skills[w] - skills[l])
            grads[w] += weight * (1.0 - p)
            grads[l] -= weight * (1.0 - p)
        sum_weights = sum(math.exp(-lambda_val * (N - 1 - idx) / N) for idx in range(N))
        l2_lambda = 0.5 * (sum_weights / N)
        for i in range(10):
            skills[i] += lr * (grads[i] - l2_lambda * skills[i])
            
    # Hybrid Mat
    weighted_wins = [[0.0]*10 for _ in range(10)]
    for idx, (w, l) in enumerate(train_duels):
        weight = math.exp(-lambda_val * (N - 1 - idx) / N)
        weighted_wins[w][l] += weight

    alpha_prior = 50.0
    P_matrix = [[0.0]*10 for _ in range(10)]
    for i in range(10):
        for j in range(10):
            if i != j:
                p_bt = sigmoid(skills[i] - skills[j])
                w_ij = weighted_wins[i][j]
                tot_ij = weighted_wins[i][j] + weighted_wins[j][i]
                P_matrix[i][j] = (w_ij + alpha_prior * p_bt) / (tot_ij + alpha_prior)
                
    # Predict Day 21
    pred_kills = [0.0]*10
    for g in day21_gauntlets:
        p1, p2 = g[0], g[1]
        pk_1 = P_matrix[p1][p2]
        pk_2 = P_matrix[p2][p1]
        
        pred_kills[p1] += pk_1
        pred_kills[p2] += pk_2
        
        survivors = {p1: pk_1, p2: pk_2}
        for chal in g[2:]:
            next_surv = {chal: 0.0}
            for surv, p_surv in survivors.items():
                p_s = P_matrix[surv][chal]
                p_c = P_matrix[chal][surv]
                pred_kills[surv] += p_surv * p_s
                pred_kills[chal] += p_surv * p_c
                
                next_surv[surv] = next_surv.get(surv, 0.0) + p_surv * p_s
                next_surv[chal] += p_surv * p_c
            survivors = next_surv
            
    # Calculate Score
    # Score = (1 / Max_AE) * sum(|pred - true|)
    # Max AE for 50 matches (200 kills) is 400.
    sum_ae = 0.0
    print("Day 21 Validation Kills:")
    for i in range(10):
        print(f"Player {i:2d} | True: {day21_true_kills[i]:6.1f} | Pred: {pred_kills[i]:6.2f} | Diff: {abs(pred_kills[i] - day21_true_kills[i]):5.2f}")
        sum_ae += abs(pred_kills[i] - day21_true_kills[i])
        
    score = sum_ae / 400.0
    print(f"\nSum of Absolute Errors: {sum_ae:.4f}")
    print(f"Estimated Validation Score: {score:.5f}")

if __name__ == "__main__":
    main()
