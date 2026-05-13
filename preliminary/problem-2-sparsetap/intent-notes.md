# Intent Notes: SparseTap

The problem is Learning Parity with Noise in disguise.

Organizer-intended path:

- Convert windows of previous bits into LPN samples `(a, b)`.
- Treat hidden taps as the sparse binary secret vector.
- Use XOR reduction such as BKW to trade sample count for noise reduction.

My original path:

- Randomly sample assumed-clean constraints.
- Solve candidates over GF(2).
- Validate candidates against remaining noisy constraints.

The organizer review explicitly notes that this RANSAC-style approach was not the intended route, but works because the problem scale was friendly.
