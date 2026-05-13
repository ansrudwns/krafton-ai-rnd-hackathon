import sys
import time
import random

def solve_gf2_bits(A_list, B_int):
    A = A_list[:]
    B = B_int
    for col in range(64):
        pivot = -1
        for r in range(col, 64):
            if (A[r] >> col) & 1:
                pivot = r
                break
        if pivot == -1: return None
        if pivot != col:
            A[col], A[pivot] = A[pivot], A[col]
            if ((B >> col) & 1) != ((B >> pivot) & 1):
                B ^= (1 << col) | (1 << pivot)
        for r in range(64):
            if r != col and ((A[r] >> col) & 1):
                A[r] ^= A[col]
                b_col_bit_val = (B >> col) & 1
                if b_col_bit_val: B ^= (1 << r)
    return B

data_path = r"c:\Users\m8686\Desktop\KRAFTON\DAY2_data.txt"
with open(data_path, 'r') as f:
    lines = [line.strip() for line in f if line.strip()]

W_max = 64
N = len(lines[0])
eqs_A = []
eqs_B = []

for line in lines:
    bits = [int(c) for c in line]
    for n in range(W_max, N):
        row_a = 0
        for k in range(1, W_max+1):
            if bits[n-k]:
                row_a |= (1 << (k-1))
        eqs_A.append(row_a)
        eqs_B.append(bits[n])

M = len(eqs_A)
random.seed(1337)

def check_solution(x, dataset_X, dataset_Y):
    matches = 0
    for i in range(len(dataset_X)):
        val = bin(dataset_X[i] & x).count('1') % 2
        if val == dataset_Y[i]: matches += 1
    return matches / len(dataset_X)

offsets = None
for it in range(1000000):
    indices = random.sample(range(M), 64)
    A_sub = [eqs_A[i] for i in indices]
    B_sub = 0
    for idx, row in enumerate(indices):
        if eqs_B[row]: B_sub |= (1 << idx)
    x = solve_gf2_bits(A_sub, B_sub)
    if x is not None:
        test_idx = random.sample(range(M), 500)
        sample_A = [eqs_A[i] for i in test_idx]
        sample_B = [eqs_B[i] for i in test_idx]
        match_rate = check_solution(x, sample_A, sample_B)
        if match_rate > 0.75:
            offsets = []
            for i in range(64):
                if (x >> i) & 1: offsets.append(i + 1)
            print("FOUND:", offsets)
            break

if offsets:
    prefix = "0000010100011010010101100101001110100011110010110011010000111010"
    seq = [int(c) for c in prefix]
    for n in range(len(prefix), 256):
        bit = 0
        for o in offsets: bit ^= seq[n - o]
        seq.append(bit)
    prediction = ''.join(str(x) for x in seq)
    print("ANSWER:")
    print(prediction[64:])
else:
    print("FAIL")
