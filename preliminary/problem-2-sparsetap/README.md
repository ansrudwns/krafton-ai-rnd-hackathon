# SparseTap

Round 1 Day 2 problem. The task was to recover hidden XOR tap offsets from noisy bit sequences.

## Why This Problem Matters

SparseTap looks like a sequence-analysis problem at first, but the intended abstraction is Learning Parity with Noise (LPN). The useful signal only appears when the correct tap set is considered jointly, so simple lag correlation is expected to fail.

## Repository Layout

- `original-submission/`: contest-time code and report.
- `intent-notes.md`: organizer review based interpretation.

## Original Submission

The submitted solution used a RANSAC-like GF(2) solving strategy and numerical/gradient-based exploration. This aligns with one of the successful unintended-but-valid solution families described in the organizer review.

Key observations:

- Single-offset correlation is not enough because the parity relation depends on multiple hidden taps.
- The submitted approach searched for consistent GF(2) structure under noise instead of relying on marginal correlations.
- Numerical experiments were used as a sanity check for whether the hidden offsets produced a recoverable signal.

## Next Refined Work

Future work should add a small BKW-style baseline or a concise experiment explaining why the RANSAC approach works at the provided scale.

Planned refined version:

1. Add a short LPN/BKW explanation with the problem variables mapped to the contest statement.
2. Implement a small BKW-style baseline or pseudocode note for the intended path.
3. Compare the submitted RANSAC-style approach with the intended abstraction.
4. Document the scale assumptions that made the submitted route viable.

## Hiring Signal

This problem shows how I move from surface-level signal processing to the underlying mathematical structure, and how I compare my own working solution against the organizer's intended abstraction after the contest.
