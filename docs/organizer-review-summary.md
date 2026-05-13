# Organizer Review Summary

Source: https://kangwooklee.com/blogs/krafton_ai_hackathon_2026/blog.html

## Hiring Signal

The hackathon was designed around a tension:

- Strong candidates should use AI aggressively to attack hard problems.
- For important problems, humans still need to understand the structure deeply enough to catch plausible but wrong AI outputs.

The useful signal is not just a polished deliverable. It is the ability to steer AI: knowing what to ask next, when to override the first answer, and when to trust the agent.

## Problem Intent

| Problem | Intent |
|---|---|
| MultiplierBoard | Test whether participants understand transformer attention and can explain a multiplication construction layer by layer. |
| SparseTap | Reveal the hidden LPN structure. BKW was the intended path, while RANSAC-style GF(2) solving was an unintended but valid path at this scale. |
| Written Exam | Sanity check whether finalists truly understood the Round 1 solutions, especially attention, sampling, and LPN noise amplification. |
| BattlePredict | Force participants to discover non-stationarity and recover anonymous day-block labels before extrapolating future player skill. |
| VideoAgent | Test real agent harness design rather than naive multimodal API calls: sampling, audio/frame extraction, multi-pass reasoning, verification, and fallback design. |

## Implication For This Repository

This repository keeps the original contest attempts separate from post-contest improvements and future refined solutions. The goal is to show both contest-time execution and deeper post-hoc understanding.
