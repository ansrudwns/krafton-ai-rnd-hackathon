# BattlePredict

Round 2 Day 1 afternoon problem. The task was to predict future player kill counts in a battle royale style toy setting with non-stationary player skill.

## Repository Layout

- `original-submission/`: contest-time code and outputs.
- `post-contest-before-review/`: self-review and fixes made before reading the organizer review.
- `refined-after-review/`: planned folder for a new solution based on the organizer's intended pipeline.
- `intent-notes.md`: problem intent and next implementation plan.

## Original Direction

The original solution used Bradley-Terry style skill estimation, empirical win matrices, exponential time decay, and Markov-chain expected kills.

## Post-Contest Finding

My immediate post-contest review identified validation time-horizon mismatch and L2 regularization scaling as major weaknesses.

## Refined Direction

The organizer review suggests the central missing step is anonymous day-block recovery. A refined solution should implement a BTL-to-Hungarian-to-refit pipeline.
