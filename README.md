# KRAFTON AI R&D Hackathon Archive

KRAFTON AI R&D Hackathon 예선과 본선에서 작성한 풀이, 제출본, 대회 직후 자체 회고를 정리한 저장소입니다.

이 저장소의 목적은 단순히 제출 코드를 보관하는 것이 아니라, 각 문제에서 요구한 **AI 활용 능력**, **문제 구조 이해**, **사후 분석과 개선 방향**을 함께 보여주는 것입니다.

## For Reviewers

이 레포는 완성된 단일 서비스가 아니라, 제한 시간 안에서 AI 도구와 수학적/엔지니어링 판단을 결합해 문제를 푼 기록입니다. 따라서 코드를 볼 때는 다음 관점으로 읽는 것을 권장합니다.

- `original-submission`: 대회 시간 안에서 실제로 제출하거나 제출을 목표로 작성한 풀이
- `post-contest-before-review`: 주최자 후기 공개 전에 스스로 발견한 문제점과 개선 시도
- `refined-after-review`: 주최자 후기의 출제 의도를 반영해 새로 정리하거나 구현할 개선 방향

핵심 평가 포인트는 정답 코드 자체보다 **문제 구조를 빠르게 모델링하는 능력**, **AI 출력물을 검증하고 보정하는 능력**, **시간 제한이 있는 상황에서 실행 가능한 파이프라인을 설계하는 능력**입니다.

## Highlights

- 약 300명이 참가한 KRAFTON AI R&D Hackathon에서 본선에 진출했습니다.
- 예선에서는 `MultiplierBoard`, `SparseTap` 두 문제를 풀었습니다.
- 본선에서는 필기형 문제, `BattlePredict`, `VideoAgent` 문제를 수행했습니다.
- 주최자 후기의 출제 의도를 바탕으로 각 문제의 intent note를 정리했습니다.
- 대회 당시 제출본과 대회 직후 자체 개선본을 구분했습니다.

## Repository Structure

```text
.
├─ docs/
│  ├─ organizer-review-summary.md
│  ├─ problem-intent-analysis.md
│  └─ my-retrospective.md
├─ preliminary/
│  ├─ problem-1-multiplierboard/
│  └─ problem-2-sparsetap/
└─ finals/
   ├─ day1/
   │  ├─ problem-1-written-exam/
   │  └─ problem-2-battlepredict/
   └─ day2/
      └─ videoagent/
```

## Version Policy

각 문제 폴더는 가능한 경우 아래 버전 구분을 따릅니다.

| Version | Meaning |
|---|---|
| `original-submission` | 대회 제한 시간 안에서 작성하거나 제출한 풀이 |
| `post-contest-before-review` | 주최자 후기 블로그를 보기 전, 대회 직후 자체 분석으로 개선한 풀이 |
| `refined-after-review` | 주최자 후기와 출제 의도를 읽은 뒤 새로 보강할 예정인 풀이 |

`refined-after-review` 폴더는 아직 계획 단계인 문제도 있습니다. 사후 개선물이 당시 제출물처럼 보이지 않도록 시점을 명확히 분리했습니다.

## Problem Map

| Stage | Problem | Folder | Focus |
|---|---|---|---|
| Preliminary Day 1 | MultiplierBoard | `preliminary/problem-1-multiplierboard` | 최소 파라미터 Transformer로 6-bit binary multiplication 수행 |
| Preliminary Day 2 | SparseTap | `preliminary/problem-2-sparsetap` | LPN 형태의 noisy XOR tap recovery |
| Finals Day 1 Problem 1 | Written Exam | `finals/day1/problem-1-written-exam` | Attention, sampling, LPN 수학 이해도 점검 |
| Finals Day 1 Problem 2 | BattlePredict | `finals/day1/problem-2-battlepredict` | Non-stationary skill rating, hidden day-block recovery |
| Finals Day 2 Problem 3 | VideoAgent | `finals/day2/videoagent` | 15분 내 20개 비디오 QA를 처리하는 multimodal agent harness |

## What This Shows

| Area | Evidence in this repo |
|---|---|
| AI-assisted problem solving | Transformer 설계, LPN 복원, 시계열 skill 추정, 멀티모달 QA harness를 각 문제 구조에 맞게 분해 |
| Mathematical modeling | GF(2) solving, Bradley-Terry skill model, Markov-chain expected kills, anonymous block recovery 방향 분석 |
| LLM/Agent engineering | VideoAgent에서 downsampling, chunking, map-reduce, retry, timeout, incremental backup 전략 설계 |
| Post-hoc analysis | 제출본과 사후 개선본을 분리하고, 주최자 후기 기반으로 의도와 한계를 재해석 |

## Recommended Reading Order

1. `docs/organizer-review-summary.md`: 해커톤이 평가하려 한 역량 요약
2. `docs/problem-intent-analysis.md`: 문제별 출제 의도와 내 풀이의 위치
3. `finals/day2/videoagent/README.md`: 제한 시간형 멀티모달 agent 설계 사례
4. `finals/day1/problem-2-battlepredict/README.md`: 시계열 예측 문제의 모델링과 사후 교정
5. `preliminary/problem-2-sparsetap/README.md`: LPN 문제에서 의도된 풀이와 제출 풀이의 차이

## Organizer Review

주최자 후기 원문:
https://kangwooklee.com/blogs/krafton_ai_hackathon_2026/blog.html

핵심 메시지는 다음과 같습니다.

> AI를 적극적으로 사용하되, AI가 그럴듯하게 틀릴 때 사람이 문제 구조를 이해하고 개입할 수 있어야 한다.

이 관점에서 각 문제의 `intent-notes.md`에는 출제 의도, 내 풀이의 위치, 향후 개선 방향을 정리했습니다.

## Problem Summaries

### MultiplierBoard

6-bit binary multiplication을 수행하는 작은 Transformer 구조를 설계하는 문제입니다. 제출본은 2-layer Transformer, parameter count, weight tying, 학습 가능성 분석을 포함합니다.

### SparseTap

Noisy XOR sequence에서 숨겨진 tap offset을 찾는 문제입니다. 주최자 의도는 LPN/BKW였고, 제출본은 RANSAC-style GF(2) solving과 numerical exploration을 사용했습니다.

### Written Exam

Round 1 풀이를 실제로 이해했는지 확인하는 no-device exam입니다. 문제 전문은 공개하지 않고, 안전한 요약과 학습 노트만 둡니다.

### BattlePredict

시간에 따라 변하는 player skill을 바탕으로 미래 kill count를 예측하는 문제입니다. 원본 제출은 Bradley-Terry, empirical matrix, Markov-chain expected kills를 활용했고, 대회 직후 회고에서는 validation horizon mismatch와 regularization scaling 문제를 분석했습니다.

### VideoAgent

20개 비디오와 20개 prompt를 제한 시간 안에 처리하는 multimodal QA agent 문제입니다. 원본 제출은 FFmpeg downsampling, chunking, Gemini Map-Reduce, incremental answer backup을 사용했습니다. 대회 직후 개선본은 timeout, async cleanup, retry 구조를 보강하는 방향으로 정리했습니다.

## What Is Excluded

GitHub 공개에 적합하지 않거나 용량이 큰 파일은 제외했습니다.

- Contest videos and sample videos (`*.mp4`)
- Submission archives (`*.zip`)
- Environment files (`.env`)
- Virtual environments (`venv/`, `.venv/`)
- Python caches (`__pycache__/`)
- Large PDFs and non-public raw assets

필요한 경우, 제외된 데이터의 역할은 각 문제 README에서 설명합니다.

## Next Steps

- `BattlePredict/refined-after-review`: organizer intent에 맞춘 BTL → Hungarian assignment → refit → extrapolate pipeline 구현
- `VideoAgent/refined-after-review`: small eval set, verification pass, timeout/retry/cancel-safe harness 구현
- `SparseTap`: BKW baseline 또는 RANSAC success condition 분석 추가
- `MultiplierBoard`: hand-proof를 layer-by-layer로 더 명확하게 정리
