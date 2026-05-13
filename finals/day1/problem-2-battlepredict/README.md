# BattlePredict

Round 2 Day 1 afternoon problem. The task was to predict future player kill counts in a battle royale style toy setting with non-stationary player skill.

## Why This Problem Matters

The surface task is a prediction problem, but the real difficulty is that player skill changes over time and part of the training data is anonymized by day. A solution has to model pairwise combat outcomes, recover hidden temporal structure, and extrapolate skill trends into future gauntlets.

## Repository Layout

- `original-submission/`: contest-time code and outputs.
- `post-contest-before-review/`: self-review and fixes made before reading the organizer review.
- `refined-after-review/`: planned folder for a new solution based on the organizer's intended pipeline.
- `intent-notes.md`: problem intent and next implementation plan.

## Original Direction

The original solution used Bradley-Terry style skill estimation, empirical win matrices, exponential time decay, and Markov-chain expected kills.

What worked:

- Pairwise battle outcomes were converted into a skill-estimation problem.
- Expected kills were computed analytically through the gauntlet order instead of simulated noisily.
- Empirical win statistics were blended with a model-based prior to stabilize sparse matchups.

## Post-Contest Finding

My immediate post-contest review identified validation time-horizon mismatch and L2 regularization scaling as major weaknesses.

What failed:

- The validation split did not match the long-horizon forecast required by the hidden test period.
- Regularization strength was not scaled consistently when time-decay weights changed.
- The original pipeline underused the anonymous day blocks, treating them more like extra evidence than recoverable temporal structure.

## Refined Direction

The organizer review suggests the central missing step is anonymous day-block recovery. A refined solution should implement a BTL-to-Hungarian-to-refit pipeline.

Planned refined version:

1. Fit Bradley-Terry skill vectors for labeled days and anonymous blocks.
2. Estimate expected skill trajectories from the labeled anchor days.
3. Match anonymous blocks to candidate days with a Hungarian assignment step.
4. Refit skill trends using recovered day labels.
5. Predict days 22-50 and evaluate sensitivity to assignment or regularization choices.

## Hiring Signal

This problem is useful as a portfolio artifact because it shows the difference between a plausible model and a problem-aware model. The refined direction demonstrates how I diagnose a wrong abstraction, identify the missing latent structure, and redesign the pipeline around it.
