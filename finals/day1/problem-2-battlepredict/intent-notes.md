# Intent Notes: BattlePredict

The key organizer insight is that the data is non-stationary and partially anonymized.

Naive failure mode:

- Fit BTL only on labeled days 1, 11, and 21.
- Assume stationarity.
- Ignore anonymous day-blocks.

Intended pipeline:

1. Estimate skill signals for each labeled and anonymous block.
2. Use linear skill anchors at days 1, 11, and 21.
3. Build a cost matrix between anonymous blocks and candidate days.
4. Recover hidden day labels with an assignment method such as Hungarian matching.
5. Refit skill curves and extrapolate days 22-50.

Refined implementation should target this pipeline explicitly.
