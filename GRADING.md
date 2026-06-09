# Grading guide — where each rubric item lives

A direct map from every rubric checkbox to where it is in this repo, so each
item is easy to find and verify.

**Quick links:** [Live demo](https://HanfuZhao781-car-recognition.hf.space) ·
[Report](TECHNICAL_REPORT.md) · [Pitch](PITCH.md) · [Models](models/) ·
[Pull Requests](https://github.com/hanfuzhao/Module1-CarRecognition/pulls)

## Project Topic & Originality
| Rubric item | Where to find it |
|---|---|
| Topic clearly defined & relevant to CV | [README](README.md) intro · [Report §1](TECHNICAL_REPORT.md) |
| New work (not reused) | [Report §1](TECHNICAL_REPORT.md); built from scratch (see PR history) |
| Real-world relevance / creativity | [Report §1](TECHNICAL_REPORT.md) (consumer car-ID use case) |
| Originality statement | [README → Originality](README.md) · [Report §3](TECHNICAL_REPORT.md) · [Pitch §2](PITCH.md) |

## Modeling Requirements
| Rubric item | Where to find it |
|---|---|
| Naive baseline implemented | `scripts/model.py` → `NaiveBaseline` · weights `models/naive_baseline_majority_class.pkl` |
| Classical (non-DL) ML model | `scripts/model.py` → `ClassicalModel` · weights `models/classical_hog_linear_svm.pkl` |
| Neural-net / deep-learning model | `scripts/model.py` → `DeepModel` · weights `models/deep_resnet50_features_mlp_head.pt` |
| All three present in repo | `scripts/model.py` + [`models/`](models/) |
| Docs on where each model lives | [README results table](README.md) + [models/README.md](models/README.md) |

## Experimentation & Analysis
| Rubric item | Where to find it |
|---|---|
| ≥1 focused experiment conducted | [Report §8](TECHNICAL_REPORT.md) (corruption robustness + confidence gating) |
| Well-motivated | [Report §8.1](TECHNICAL_REPORT.md) |
| Setup clearly described | [Report §8.2](TECHNICAL_REPORT.md) |
| Results meaningfully interpreted | [Report §8.4 / §8.6](TECHNICAL_REPORT.md) |
| Experiment informs modeling/system design | [Report §8.6](TECHNICAL_REPORT.md) (confidence-gate deploy policy → `main.py`) |

## Interactive Application
| Rubric item | Where to find it |
|---|---|
| Publicly accessible URL | https://HanfuZhao781-car-recognition.hf.space |
| Live for 1 week | keep-alive workflow [`.github/workflows/keep-alive.yml`](.github/workflows/keep-alive.yml) pings every 6h |
| Inference only (no training) | `main.py` (loads the trained head only) |
| Runs successfully when graded | live demo above; `GET /health` returns `{"status":"ok"}` |
| Model inference end-to-end | `main.py` → `/predict` (upload → top-5 + confidence) |
| Usable, product-level UX | `templates/index.html`, `static/css/style.css`, `static/js/app.js` |
| Stable, no broken assets | served by the Space; CSS/JS/health all return 200 |

## Written Report — [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)
| Rubric item | Section |
|---|---|
| Problem Statement | §1 |
| Data Sources | §2 |
| Related Work (lit review) | §3 |
| Evaluation Strategy & Metrics (justified) | §4 |
| Modeling Approach | §5 |
| Data Processing Pipeline + rationale | §5.1 |
| Hyperparameter Tuning Strategy | §5.2 |
| Model Evaluations (all 3) | §6.1 |
| Quantitative comparison across models | §6.1 |
| Visualizations | §6 / §8 + [`data/outputs/*.png`](data/outputs/) |
| Confusion matrices | §6.2 + `data/outputs/confusion_matrix.png` |
| ≥5 specific mispredictions | §7 |
| Root causes explained | §7 |
| Concrete mitigation strategies | §7 |
| Experimental plan | §8.2 |
| Results reported | §8.3 / §8.5 |
| Interpretation provided | §8.4 / §8.6 |
| Actionable recommendations | §8.6 |
| Conclusions | §10 |
| Future Work (another semester) | §11 |
| Commercial Viability Statement | §12 |
| Ethics Statement | §13 |

## In-Class Pitch — [PITCH.md](PITCH.md)
| Rubric item | Where to find it |
|---|---|
| Problem & motivation | Pitch §1 |
| Approach overview | Pitch §2 |
| Live demo (or video link) | Pitch §3 + [live demo](https://HanfuZhao781-car-recognition.hf.space) |
| Results / insights / key findings | Pitch §4 |
| Respects 5-min limit | Pitch is timed (~4m40s) |

## Code & Repository — Git Best Practices
| Rubric item | Where to find it |
|---|---|
| Branches / forks used | [Pull Requests](https://github.com/hanfuzhao/Module1-CarRecognition/pulls) — every change on its own branch |
| PRs made (individual project) | 20+ merged PRs, each a single feature |
| PR reviews before merge | review comment on each PR before merge |

## Code Quality & Practices
| Rubric item | Where to find it |
|---|---|
| No executable code outside functions / `__main__` | all modules; logic is in functions/classes |
| Modularized into functions and classes | `scripts/` (`data.py`, `model.py`, `experiment.py`, `eda.py`, `make_dataset.py`) |
| Notebooks only in `notebooks/` | [`notebooks/`](notebooks/) (none committed) |

## Reproducibility & Documentation
| Rubric item | Where to find it |
|---|---|
| Code well organized | repo layout in [README](README.md) |
| All files to run included | models committed in [`models/`](models/); deps in `requirements.txt` |
| Runs with provided instructions | [README → Quick start](README.md); `python main.py` runs the app from the committed model |
| Well commented / docstrings | every module has docstrings + comments |
| Descriptive README | [README.md](README.md) |
