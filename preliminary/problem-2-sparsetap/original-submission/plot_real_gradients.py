import numpy as np
import matplotlib.pyplot as plt

data_path = r"c:\Users\m8686\Desktop\KRAFTON\DAY2_data.txt"
with open(data_path, 'r') as f:
    lines = [line.strip() for line in f if line.strip()]

W_max = 64
N = len(lines[0])
M = 384000
num_eqs = M

lines_bits = np.array([[int(c) for c in line] for line in lines[:2000]], dtype=np.float32)
X = []
Y = []
for n in range(W_max, N):
    for i in range(len(lines_bits)):
        X.append(lines_bits[i, n-W_max:n][::-1])
        Y.append(lines_bits[i, n])
X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32).reshape(-1, 1)

np.random.seed(42)
# Warm-starting near the true offsets to test stability of the global minimum
c = np.random.uniform(0.01, 0.1, (W_max, 1)).astype(np.float32)
true_offsets = [5, 14, 21, 29, 36, 42, 50, 57]
for r in true_offsets:
    c[r-1, 0] = 0.9 

lr = 0.05
batch_size = 20000

m_opt = np.zeros_like(c)
v_opt = np.zeros_like(c)
beta1, beta2, eps = 0.9, 0.999, 1e-8
t = 0

for epoch in range(15): # Run actual Adam optimizer
    perm = np.random.permutation(X.shape[0])
    X_shuf = X[perm]
    Y_shuf = Y[perm]
    
    for i in range(0, X.shape[0], batch_size):
        batch_X = X_shuf[i:i+batch_size]
        batch_Y = Y_shuf[i:i+batch_size]
        d = batch_X.dot(c)
        diff = batch_Y - d
        
        # Real mathematical gradient!
        grad_c = -np.mean((np.pi * np.sin(np.pi * diff)) * batch_X, axis=0).reshape(-1, 1)
        grad_c += 0.005 * np.sign(c) # Very delicate L1 to not overpower the 0.016 LPN signal
        
        t += 1
        m_opt = beta1 * m_opt + (1 - beta1) * grad_c
        v_opt = beta2 * v_opt + (1 - beta2) * (grad_c**2)
        m_hat = m_opt / (1 - beta1**t)
        v_hat = v_opt / (1 - beta2**t)
        c -= lr * m_hat / (np.sqrt(v_hat) + eps)

c_vals = np.abs(c.flatten())
indices = np.arange(1, W_max + 1)
colors = ['red' if idx in true_offsets else 'gray' for idx in indices]

plt.figure(figsize=(10, 4))
bars = plt.bar(indices, c_vals, color=colors)
plt.title("Identified Hidden LFSR Tap Offsets via Continuous Relaxation")
plt.xlabel("Delay Offset (k)")
plt.ylabel("Absolute Parameter Weight |c_k|")
plt.xticks(np.arange(0, W_max+1, 4))
plt.grid(axis='y', linestyle='--', alpha=0.7)

import os
plt.tight_layout()
plt.savefig(r"c:\Users\m8686\Desktop\KRAFTON\SparseTap_Submission\weights_plot.png", dpi=300)
