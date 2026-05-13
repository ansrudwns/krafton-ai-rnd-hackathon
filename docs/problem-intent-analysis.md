# Problem Intent Analysis

## Round 1

### MultiplierBoard

The problem asks for the smallest Transformer that can multiply two 6-bit binary numbers. The intended signal is not just model accuracy. It is whether the participant can reason about attention heads, MLP computation, carry propagation, and parameter efficiency.

### SparseTap

The surface task looks like signal processing or LFSR cracking, but the intended abstraction is Learning Parity with Noise. The official intended path points toward BKW-style XOR reduction. My submitted approach used a RANSAC/GF(2)-style route, which the organizer review describes as unintended but valid at the specific scale.

## Round 2

### Written Exam

The written exam bridges Round 1 with fundamentals: attention by hand, sampling strategies, and LPN noise amplification. It checks whether a strong AI-assisted submission was backed by actual understanding.

### BattlePredict

The central trick is non-stationary skill. A naive BTL fit on labeled days gives a plausible but wrong answer. The intended pipeline is closer to:

1. Estimate per-block skill signals.
2. Recover anonymous day labels.
3. Use a Hungarian assignment-style matching step.
4. Refit and extrapolate days 22-50.

### VideoAgent

The task is not solved by sending entire videos to a multimodal model. The intended engineering problem is harness design: frame/audio sampling, chunking, multi-pass reasoning, verification, and robust fallback behavior under time pressure.
