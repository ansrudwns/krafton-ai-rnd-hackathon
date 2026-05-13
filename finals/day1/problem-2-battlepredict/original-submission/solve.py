import csv
import math

# Sigmoid function for BT
def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

def main():
    csv_file = '[KRAFTON AI R&D HACKATHON] DAY1 P2 dataset.csv'
    
    train_duels = []
    test_gauntlets = []
    
    wins = [[0]*10 for _ in range(10)]
    
    # 1. Parse Data
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            events = row['events'].strip().split()
            # If day is 22..50 or events does not contain '>', it's a test gauntlet
            day = row['day']
            is_train = True
            if day != '?' and int(day) >= 22:
                is_train = False
            
            # Additional check: test gauntlets only have space separated numbers without '>'
            if not any('>' in e for e in events):
                is_train = False
                
            if is_train:
                for e in events:
                    if '>' in e:
                        w, l = e.split('>')
                        w, l = int(w), int(l)
                        train_duels.append((w, l))
                        wins[w][l] += 1
            else:
                test_gauntlets.append([int(x) for x in events])
                
    print(f"Total train duels: {len(train_duels)}")
    print(f"Total test gauntlets: {len(test_gauntlets)}")
    
    # 2. Fit Bradley-Terry Model
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
            prob = sigmoid(skills[w] - skills[l])
            grads[w] += weight * (1.0 - prob)
            grads[l] -= weight * (1.0 - prob)
            
        sum_weights = sum(math.exp(-lambda_val * (N - 1 - idx) / N) for idx in range(N))
        l2_lambda = 0.5 * (sum_weights / N)
        for i in range(num_players):
            # Update with rescaled L2 regularization
            skills[i] += lr * (grads[i] - l2_lambda * skills[i])

    print("\nBradley-Terry Learned Skills (Time-Weighted):")
    for i, s in enumerate(skills):
        print(f"Player {i}: {s:.4f}")
        
    # 3. Create Hybrid Probability Matrix
    alpha_prior = 50.0 # Optimized Weight for BT prior
    P_matrix = [[0.0]*10 for _ in range(10)]
    weighted_wins = [[0.0]*10 for _ in range(10)]
    
    for idx, (w, l) in enumerate(train_duels):
        weight = math.exp(-lambda_val * (N - 1 - idx) / N)
        weighted_wins[w][l] += weight
    
    for i in range(10):
        for j in range(10):
            if i != j:
                p_bt = sigmoid(skills[i] - skills[j])
                w_ij = weighted_wins[i][j]
                tot_ij = weighted_wins[i][j] + weighted_wins[j][i]
                P_matrix[i][j] = (w_ij + alpha_prior * p_bt) / (tot_ij + alpha_prior)

    # 4. Simulate Test Gauntlets analytically
    expected_kills = [0.0] * 10
    
    for gauntlet in test_gauntlets:
        # State maintains probability of being the survivor
        p1 = gauntlet[0]
        p2 = gauntlet[1]
        
        pk_win_1 = P_matrix[p1][p2]
        pk_win_2 = P_matrix[p2][p1]
        
        expected_kills[p1] += pk_win_1
        expected_kills[p2] += pk_win_2
        
        survivors = {p1: pk_win_1, p2: pk_win_2}
        
        for challenger in gauntlet[2:]:
            next_survivors = {challenger: 0.0}
            for surv, p_surv in survivors.items():
                p_surv_wins = P_matrix[surv][challenger]
                p_chal_wins = P_matrix[challenger][surv]
                
                expected_kills[surv] += p_surv * p_surv_wins
                expected_kills[challenger] += p_surv * p_chal_wins
                
                # accumulate survivor probabilities
                next_survivors[surv] = next_survivors.get(surv, 0.0) + p_surv * p_surv_wins
                next_survivors[challenger] += p_surv * p_chal_wins
                
            survivors = next_survivors
            
    print("\nFinal Predicted Total Kills (Days 22-50):")
    total_calculated = 0
    with open('predictions.txt', 'w') as f:
        for i, kills in enumerate(expected_kills):
            res = f"user_{i}: {kills:.4f}"
            print(res)
            f.write(res + "\n")
            total_calculated += kills
            
        f.write(f"\nChecksum: {total_calculated:.4f}\n")
    print(f"\nChecksum ({total_calculated:.4f}) written to predictions.txt")

if __name__ == "__main__":
    main()
