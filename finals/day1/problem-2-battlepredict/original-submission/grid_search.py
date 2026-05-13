import csv
import math

def sigmoid(x):
    if x >= 0: return 1.0 / (1.0 + math.exp(-x))
    else: return math.exp(x) / (1.0 + math.exp(x))

def reconstruct_gauntlet(events):
    duels = []
    for e in events:
        w, l = e.split('>')
        duels.append((int(w), int(l)))
    w1, l1 = duels[0]
    gauntlet = [w1, l1]
    last_winner = w1
    for i in range(1, 4):
        w, l = duels[i]
        if w == last_winner: new_player = l
        elif l == last_winner: new_player = w
        else: new_player = l
        gauntlet.append(new_player)
        last_winner = w
    return gauntlet

def main():
    csv_file = '[KRAFTON AI R&D HACKATHON] DAY1 P2 dataset.csv'
    train_duels = []
    
    # We will test on Days 12..21.
    # Day 1..11 is train.
    test_gauntlets = []
    test_true_kills = [0.0]*10
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        # We need to manually track line numbers to know where we are because `day` has '?'
        # 1-50: Day 1
        # 51-500: Hidden Day 2-10 (450 matches)
        # 501-550: Day 11 (50 matches)
        # 551-1000: Hidden Day 12-20 (450 matches)
        # 1001-1050: Day 21 (50 matches)
        # 1051-2500: Day 22-50
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if '>' not in row['events']: continue
            events = row['events'].strip().split()
            
            # i goes from 0 to 2499.
            # 0 to 549 -> Day 1 to 11
            if i < 550:
                for e in events:
                    w, l = e.split('>')
                    train_duels.append((int(w), int(l)))
            elif i < 1050:
                # 550 to 1049 -> Day 12 to 21
                for e in events:
                    w, _ = e.split('>')
                    test_true_kills[int(w)] += 1.0
                test_gauntlets.append(reconstruct_gauntlet(events))
                    
    num_players = 10
    epochs = 1500
    lr = 0.01
    N = len(train_duels) # This should be 550 * 4 = 2200
    
    # Testing over a 10-day period (500 matches = 2000 kills)
    # Total sum of errors Max is 4000.
    
    lambdas = [0.0, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0]
    alphas = [0.1, 1.0, 3.0, 5.0, 10.0, 20.0, 50.0]
    
    best_score = float('inf')
    best_params = (0, 0)
    
    print(f"Starting Grid Search. Wait...")
    for lam in lambdas:
        sum_weights = sum(math.exp(-lam * (N - 1 - idx) / N) for idx in range(N))
        l2_lambda = 0.5 * (sum_weights / N)  # Rescaled L2 penalty
        
        # Fit BT
        skills = [0.0] * num_players
        for epoch in range(epochs):
            grads = [0.0] * num_players
            for idx, (w, l) in enumerate(train_duels):
                weight = math.exp(-lam * (N - 1 - idx) / N)
                p = sigmoid(skills[w] - skills[l])
                grads[w] += weight * (1.0 - p)
                grads[l] -= weight * (1.0 - p)
            for i in range(10):
                skills[i] += lr * (grads[i] - l2_lambda * skills[i])
                
        # Prep base weighted wins
        weighted_wins = [[0.0]*10 for _ in range(10)]
        for idx, (w, l) in enumerate(train_duels):
            weight = math.exp(-lam * (N - 1 - idx) / N)
            weighted_wins[w][l] += weight
            
        for alf in alphas:
            P_matrix = [[0.0]*10 for _ in range(10)]
            for i in range(10):
                for j in range(10):
                    if i != j:
                        p_bt = sigmoid(skills[i] - skills[j])
                        w_ij = weighted_wins[i][j]
                        tot_ij = weighted_wins[i][j] + weighted_wins[j][i]
                        P_matrix[i][j] = (w_ij + alf * p_bt) / (tot_ij + alf)
                        
            # Predict
            pred_kills = [0.0]*10
            for g in test_gauntlets:
                p1, p2 = g[0], g[1]
                pk_1, pk_2 = P_matrix[p1][p2], P_matrix[p2][p1]
                pred_kills[p1] += pk_1
                pred_kills[p2] += pk_2
                surv = {p1: pk_1, p2: pk_2}
                for c in g[2:]:
                    next_s = {c: 0.0}
                    for s, ps in surv.items():
                        ps_wins = P_matrix[s][c]
                        pc_wins = P_matrix[c][s]
                        pred_kills[s] += ps * ps_wins
                        pred_kills[c] += ps * pc_wins
                        next_s[s] = next_s.get(s, 0.0) + ps * ps_wins
                        next_s[c] += ps * pc_wins
                    surv = next_s
                    
            sum_ae = sum(abs(pred_kills[i] - test_true_kills[i]) for i in range(10))
            score = sum_ae / 4000.0  # Max error is 2 * 2000 = 4000
            
            if score < best_score:
                best_score = score
                best_params = (lam, alf)
                print(f"New Best! Lambda: {lam}, Alpha: {alf} -> Score: {score:.5f} (SAE: {sum_ae:.2f})")

    print(f"\nOptimal Param: Lambda={best_params[0]}, Alpha={best_params[1]} with Score={best_score:.5f}")

if __name__ == "__main__":
    main()
