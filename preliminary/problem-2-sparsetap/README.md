# SparseTap

Round 1 Day 2 problem. The task was to recover hidden XOR tap offsets from noisy bit sequences.

## Repository Layout

- `original-submission/`: contest-time code and report.
- `intent-notes.md`: organizer review based interpretation.

## Original Submission

The submitted solution used a RANSAC-like GF(2) solving strategy and numerical/gradient-based exploration. This aligns with one of the successful unintended-but-valid solution families described in the organizer review.

## Next Refined Work

Future work should add a small BKW-style baseline or a concise experiment explaining why the RANSAC approach works at the provided scale.
