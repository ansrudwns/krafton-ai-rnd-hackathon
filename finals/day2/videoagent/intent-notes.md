# Intent Notes: VideoAgent

The task was designed so that naive multimodal API calls would fail on hard instances.

Important design choices:

- Build a video QA harness rather than simply call a SOTA model.
- Decide frame sampling, audio extraction, chunking, model selection, and verification strategy.
- Construct an evaluation set if using meta-harness optimization.
- Add fallback heuristics and pass-by-pass cross-checking.
- Keep the system robust under strict time pressure.

Refined implementation should use small publishable sample assets or synthetic examples, because the original contest videos are large and may not be appropriate to publish.
