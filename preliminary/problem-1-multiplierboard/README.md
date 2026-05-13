# MultiplierBoard

Round 1 Day 1 problem. The goal was to design a compact Transformer architecture that can multiply two 6-bit binary numbers.

## Repository Layout

- `original-submission/`: contest-time code and reports.
- `intent-notes.md`: organizer review based interpretation.

## Original Submission

The submitted approach used a 2-layer Transformer structure, parameter counting, weight tying, and a trained 16-dimensional configuration that reached high accuracy under the fixed protocol.

## Next Refined Work

Future work should make the hand-proof clearer: which attention head routes which bits, what each MLP block computes, and why carry propagation requires at least two layers.
