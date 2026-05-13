# VideoAgent

Round 2 Day 2 problem. The task was to answer 20 video questions within a strict 15-minute submission window after receiving the hidden videos and prompts.

## Repository Layout

- `original-submission/`: contest-time VideoAgent V3 files.
- `post-contest-before-review/`: self-improved version made before reading the organizer review.
- `refined-after-review/`: planned folder for a new solution based on organizer intent.
- `intent-notes.md`: problem intent and next implementation plan.

## Original Direction

The original approach used FFmpeg downsampling, chunking, Gemini-based Map-Reduce, and incremental answer backup.

## Post-Contest Finding

The immediate self-review identified missing hard timeouts, zombie async tasks, and retry logic that reused exhausted API keys.

## Refined Direction

The organizer review emphasizes agent harness design: frame/audio sampling, evaluation set construction, multi-pass reasoning, verification, and fallback strategies.
