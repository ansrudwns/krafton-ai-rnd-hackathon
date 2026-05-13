# KRAFTON AI R&D Hackathon

KRAFTON AI R&D Hackathon 예선과 본선에서 작성한 풀이, 제출본, 대회 직후 자체 회고를 정리한 저장소입니다.

이 저장소는 세 가지 버전을 구분합니다.

| Version | Meaning |
|---|---|
| `original-submission` | 대회 제한 시간 안에서 작성하거나 제출한 풀이 |
| `post-contest-before-review` | 주최자 후기 블로그를 보기 전, 대회 직후 자체 분석으로 개선한 풀이 |
| `refined-after-review` | 주최자 후기와 출제 의도를 읽은 뒤 새로 보강할 예정인 풀이 |

대용량 비디오, 제출용 압축 파일, `.env`, 가상환경, 비공개 가능성이 있는 원본 데이터는 제외했습니다.

## Problem Map

| Stage | Problem | Folder | Focus |
|---|---|---|---|
| Preliminary Day 1 | MultiplierBoard | `preliminary/problem-1-multiplierboard` | 최소 파라미터 Transformer로 6-bit binary multiplication 수행 |
| Preliminary Day 2 | SparseTap | `preliminary/problem-2-sparsetap` | LPN 형태의 noisy XOR tap recovery |
| Finals Day 1 Problem 1 | Written Exam | `finals/day1/problem-1-written-exam` | Attention, sampling, LPN 수학 이해도 점검 |
| Finals Day 1 Problem 2 | BattlePredict | `finals/day1/problem-2-battlepredict` | Non-stationary skill rating, hidden day-block recovery |
| Finals Day 2 Problem 3 | VideoAgent | `finals/day2/videoagent` | 15분 내 20개 비디오 QA를 처리하는 multimodal agent harness |

## Organizer Review

주최자 후기 원문:
https://kangwooklee.com/blogs/krafton_ai_hackathon_2026/blog.html

핵심 메시지는 "AI를 적극적으로 사용하되, AI가 그럴듯하게 틀릴 때 사람이 문제 구조를 이해하고 개입할 수 있어야 한다"는 것입니다. 각 문제의 `intent-notes.md`에는 이 관점에서 문제 의도와 내 풀이 방향을 정리합니다.

## What Is Excluded

- Contest videos and sample videos (`*.mp4`)
- Submission archives (`*.zip`)
- Environment files (`.env`)
- Virtual environments (`venv/`, `.venv/`)
- Python caches (`__pycache__/`)
- Large PDFs and non-public raw assets

Excluded files are documented when they are necessary to understand how to run or reproduce the project.
