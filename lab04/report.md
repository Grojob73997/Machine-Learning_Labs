# Lab 4: Data pipeline and quality

**Student:** G M Robi Farhan Rojob | Student number 73997 | CII5SE039CI | Group 01

## 1. Goal and problem definition

The task is binary supervised classification: predict whether an Adult record has annual income above USD 50,000. Each input contains 14 features: six integer-valued attributes (age, census final weight, education number, capital gain, capital loss and weekly working hours) and eight categorical attributes describing work, education, marital status, occupation, household relationship, race, sex and country of origin. The target is income >50K, encoded as 1; <=50K is 0. OpenML calls the target class, so a separate working view names it income while the original ingestion frame remains unchanged. The data are historical US census records, so the threshold describes that source period and is not adjusted for inflation.

Success is measured with held-out accuracy and binary F1 for >50K. Accuracy is the fraction of correct predictions; F1 = 2TP/(2TP+FP+FN) balances precision and recall for the less frequent positive class. Both metrics are needed because an always-negative classifier can appear accurate on an imbalanced dataset. The aim is a transparent LogisticRegression baseline and a controlled comparison after adding two predetermined features; the manual specifies no minimum score. The split, model settings and feature formulas are fixed before measuring either test score. The test set is not used for hyperparameter tuning or feature selection.

The engineering constraints are reproducibility, explicit data-quality reporting and no test-derived fitted statistics. Raw input must be preserved, the split must precede cleaning, and imputation, clipping limits, scaling and category vocabulary must be learned from retained training rows only. Every source row must be accounted for, transformations must be reusable on new data, and a SHA256 fingerprint must identify the exact processed bytes. The intended use is a laboratory study of data engineering. The model does not establish causes of income or suitability for decisions about individuals. The data card records historical bias, sensitive attributes and privacy limitations; the duplicate audit separately identifies dependence that a prescribed random row split may leave across train and test.

## 2. Method

### Ingestion and validation

Used sklearn.datasets.fetch_openml(name='adult', version=2, as_frame=True), OpenML ID 1590. Ingestion logs 48,842 rows and 15 columns without modifying raw values. A canonical raw CSV and the OpenML download cache are stored under the ignored root data/ directory. The original class column is only renamed on a copied view. Raw and test-frame content hashes are checked after processing.

The fixed dictionary in src/schema.py declares all 15 schema types, permitted categories (including ? for feature fields), and the manual's inclusive age 17-90, hours-per-week 1-99, capital-gain 0-99999 and capital-loss 0-4356 ranges. Extra checks cover positive fnlwgt, education-num 1-16 and agreement between education and education-num. Integral numeric values satisfy the integer schema; physical storage dtypes are reported separately. Both parser NaN and literal ? mean feature missingness. Missing target labels and unknown categories are violations. Validation is read-only, reports all failing cells and stops the run on violations; it never silently removes rows. Missing feature values are permitted and counted.

### Split and train-only cleaning

Performed train_test_split with test_size=0.30, stratify=income and random_state=42 before any cleaning or duplicate removal. The split contains 34,189 training and 14,653 test rows. Exact duplicate FULL rows, including the target, are removed only from training with keep='first', leaving 34,165 rows (24 removed). Row IDs and all split assignments are saved. Conflicting-label feature duplicates are audited, never resolved by looking at test labels.

The training-only sklearn Pipeline converts feature ? to missing, fits numeric medians and categorical modes, and learns 1st/99th percentiles on imputed training numeric values. Every original numeric feature is clipped using these fixed limits. Derived features are then computed from those cleaned, clipped values: capital_net = capital-gain - capital-loss; hours_per_age = hours-per-week/(age-16). Validation ensures age>16 and the transformer also guards division. The derived columns are scaled with the original numeric columns; original numeric columns are not removed. There is no separate clipping of derived columns. StandardScaler is fitted on train only. OneHotEncoder learns its vocabulary from train only and uses handle_unknown='ignore'. The test partition keeps every original row and label; only transform is applied to produce model inputs. 'Untouched test' means no deduplication, fitted statistics or label edits, not skipping the required train-fitted transformations.

Both comparisons use LogisticRegression(solver='lbfgs', C=1, max_iter=2000, tol=0.0001, random_state=42) and threshold 0.5. The six numeric features and fnlwgt remain predictors; fnlwgt is not used as sample_weight, so scores are unweighted record-level measures. There is no class weighting or test-driven tuning. Non-convergence is treated as an error. The final engineered transformer is saved at results/preprocessor.joblib; the baseline transformer and both classifiers are also saved.

### Label audit and reproducibility

The full raw dictionary/duplicate audit runs for reporting only and does not change any fitted model. Feature-identical rows with different labels are counted as conflict groups and affected source rows. A fixed seed-42 sample of 50 source rows is saved in audit_sample_50.csv with dictionary checks. The student's manual review worksheet is manual_review_50.csv. It is pending, not represented as completed human evidence, and reruns preserve annotations. A valid dictionary row is not proof that the person's income label is correct. Adult does not require the subjective-label 24-hour relabeling or Cohen's kappa branch of the manual.

Python, NumPy and both prescribed torch seeds use 42. CPU numerical libraries use one thread. Python and every installed distribution are printed and saved. The run exports canonical processed CSV bytes (UTF-8, LF, no index, header, %.17g floats, source IDs and labels included, retained train rows then every test row). Their SHA256 is stored beside the full run config in results/config.yaml. A second fit repeats split, deduplication, preprocessing and both models from the same immutable input; it checks split IDs, arrays, probabilities, predictions and processed SHA256. Joblib reload equivalence and targeted tests are also checked. Hash equality establishes byte identity, not label truth or fairness. Cross-version/hardware bitwise identity is not assumed; the measured environment is recorded.

## 3. Results

| Variant | Train rows | Test rows | Encoded features | Accuracy | F1 (>50K) |
|---|---|---|---|---|---|
| baseline | 34165 | 14653 | 105 | 0.853955 | 0.660102 |
| engineered | 34165 | 14653 | 107 | 0.857777 | 0.669626 |

The fixed engineered comparison changes accuracy by +0.003822 and F1 by +0.009524. This is one held-out split, not evidence of statistical significance. Both scores are reported without selecting settings from the test result. Confusion matrices, predictions, convergence iterations, class proportions and clipping counts are available under results/.

![Held-out comparison](results/feature_comparison.png)

Validation found **0 violations** across the specified schema, ranges and additional dictionary checks. Missing counts:

| Column | Parser nulls | Literal ? | Total missing |
|---|---|---|---|
| workclass | 2799 | 0 | 2799 |
| occupation | 2809 | 0 | 2809 |
| native-country | 857 | 0 | 857 |

Raw exact-duplicate excess rows: **52**. Feature-duplicate excess rows: **57**. Conflicting feature groups: **5**; affected rows: **10**. Exact duplicates removed from training: **24**. Test rows matching a retained training full row: **24**; matching features regardless of label: **27**. The manual's random row split and untouched test policy allow these overlaps. All fitted statistics are isolated from test, but the partitions cannot be claimed fully independent at the entity/duplicate level. A separately specified group split would be needed to study that limitation, and is not substituted for the required split here.

The automated dictionary checks found no impossible values in the fixed 50-row sample. **Student manual review: pending.** Follow manual_review_instructions.md and fill the preserved worksheet. The three conclusions below are draft explanations to review and rewrite in your own words before submission.

Processed dataset SHA256: `50a22304e1125a60d773b504cc25c762597e74732c244b8bd8abfbce317cd115`.

## 4. Three draft conclusions

1. Splitting before cleaning prevents test information from influencing medians, modes, clipping percentiles, scaling or category vocabularies. Persisting the fitted transformer applies exactly those training decisions to later inputs.
2. The two derived features changed held-out F1 by +0.009524 on this fixed split. A useful formula does not guarantee an improvement, and this single comparison does not justify tuning against the same test set.
3. Data quality includes missingness, duplicate dependence, label consistency and provenance, beyond valid ranges. The repeatable SHA256 identifies exact processed bytes, while the data card and pending manual review explain what that fingerprint cannot establish.

## 5. Evidence and references

- results/validation_report.md and validation_violations.csv: schema, range and missingness evidence.
- src/, config.yaml and results/preprocessor.joblib: executable pipeline and fitted state.
- data_card.md: source, license, sizes, biases, privacy and transformations.
- results/reproducibility.json, verification.txt, config.yaml and processed_dataset.sha256: repeatability evidence.
- results/manual_review_50.csv: student-owned manual review worksheet (pending).
- Lab Manual, Lab 4, physical PDF page 1 / printed page 6; global rules from Lab 1 pp.1-2.
- Vijay Janapa Reddi, Introduction to Machine Learning Systems, supplied July 31, 2025 edition: Chapter 5 §§5.2-5.5 (printed pp.206-217), Chapter 6 §§6.2-6.8 (printed pp.242-288). PDF viewer pages are printed pages +52. The companion study guide maps the readings to implementation decisions.
- [UCI Adult dataset and dictionary](https://archive.ics.uci.edu/dataset/2/adult); [DOI](https://doi.org/10.24432/C5XW20); [OpenML 1590](https://www.openml.org/d/1590).

Implementation and draft documentation were prepared with AI assistance. Review, understand and personalize the work in accordance with your course's individual-work rules. No manual inspection, attendance or personal authorship is asserted on your behalf.
