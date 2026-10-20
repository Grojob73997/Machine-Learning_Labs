# Lab 1: Environment and First System Measurements

**Student:** G M Robi Farhan Rojob  
**Student ID:** 73997

## Goal

Create a reproducible Python environment, train two baseline classifiers on the Breast Cancer Wisconsin diagnostic dataset, measure their system cost, and assess deployment suitability against Cloud, Edge, Mobile, and TinyML budgets.

## Method

- Python 3.11.8 is the required interpreter; dependencies are pinned in the repository root `requirements.txt`.
- Randomness is controlled with `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`, and `torch.cuda.manual_seed_all(42)`.
- The dataset is loaded with `load_breast_cancer(return_X_y=True)` and split 70/30 using stratification and `random_state=42`.
- Baselines are `LogisticRegression(max_iter=1000, random_state=42)` and `RandomForestClassifier(n_estimators=100, random_state=42)`.
- Training is warmed up once and measured five times; single-sample inference is measured 100 times. Medians are reported.
- Model sizes are measured from uncompressed `joblib` files. Peak memory is sampled with `memory_profiler` during training and inference.

## Results

The generated evidence is stored in `results/`:

- `baseline_accuracy.csv` — test accuracy to four decimal places.
- `measurement_table.csv` — timing, model-size, and peak-memory measurements.
- `resource_budget.csv` — deployment-budget decisions.
- `versions.txt` — Python and package versions.
- `model.joblib` — combined serialized model artifact.

### Execution note

The laboratory manual contains a dependency-resolution conflict: `mlflow==2.14.1` requires `pyarrow<16`, while the manual also pins `pyarrow==16.1.0`. The exact pins remain in `requirements.txt`; the Lab 1 experiment was executed with the core pinned packages needed by this lab.

### Baseline accuracy

| Model | Test accuracy |
|---|---:|
| Logistic Regression | 0.9474 |
| Random Forest | 0.9357 |

### System measurements

| Model | Training median (ms) | Single-sample inference median (ms) | Model size (bytes) | Model size (KB) | Peak memory (MB) |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 817.271400 | 0.108800 | 1,055 | 1.030 | 530.352 |
| Random Forest | 253.768000 | 5.093050 | 290,889 | 284.071 | 530.898 |

The training medians use five measured runs after one warm-up. Inference medians use 100 single-sample runs. Peak memory is the larger of the training and inference peak measurements.

### Deployment budget assessment

| Model | Cloud | Edge | Mobile | TinyML |
|---|---|---|---|---|
| Logistic Regression | Yes | Yes | No | No |
| Random Forest | Yes | Yes | No | No |

Both models fit Cloud and Edge latency and serialized-size limits. Neither fits Mobile because the measured peak process memory is above 256 MB. Neither fits TinyML because the measured memory is above 256 KB; the Random Forest also exceeds the 100 KB model-size limit. The full evidence is in `results/resource_budget.csv`.

## Conclusions

1. Logistic Regression achieved the higher test accuracy (0.9474 versus 0.9357), while Random Forest had the lower median training time (253.7680 ms versus 817.2714 ms).
2. The median measurements are more reliable than a single timing run because they reduce the influence of startup and scheduling noise.
3. Deployment suitability depends on all three resource dimensions—latency, serialized model size, and memory—not accuracy alone; both models fit Cloud and Edge, but neither meets the measured Mobile or TinyML budgets.
