"""Generate report/data card from measured outputs; no invented results."""
import json
from pathlib import Path

IDENTITY = "G M Robi Farhan Rojob | Student number 73997 | CII5SE039CI | Group 01"


def md_table(headers, rows):
    return "| " + " | ".join(headers) + " |\n|" + "|".join(["---"]*len(headers)) + "|\n" + "\n".join(
        "| " + " | ".join(str(c) for c in row) + " |" for row in rows)


def write_documents(lab,out,config,metrics,missing,conflicts,overlap,nraw,ntrain,ntest,digest):
    m0,m1 = metrics
    metric_table = md_table(["Variant","Train rows","Test rows","Encoded features","Accuracy","F1 (>50K)"],
        [[m["variant"],m["train_rows"],m["test_rows"],m["encoded_features"],f'{m["accuracy"]:.6f}',f'{m["f1"]:.6f}'] for m in metrics])
    missing_table = md_table(["Column","Parser nulls","Literal ?","Total missing"],
        [[r.column,r.null_count,r.question_mark_count,r.missing_total] for r in missing.itertuples() if r.missing_total])
    report = f"""# Lab 4: Data pipeline and quality

**Student:** {IDENTITY}

## 1. Goal and problem definition

The task is binary supervised classification: predict whether an Adult record has annual income above USD 50,000. Each input contains 14 features: six integer-valued attributes (age, census final weight, education number, capital gain, capital loss and weekly working hours) and eight categorical attributes describing work, education, marital status, occupation, household relationship, race, sex and country of origin. The target is income >50K, encoded as 1; <=50K is 0. OpenML calls the target class, so a separate working view names it income while the original ingestion frame remains unchanged. The data are historical US census records, so the threshold describes that source period and is not adjusted for inflation.

Success is measured with held-out accuracy and binary F1 for >50K. Accuracy is the fraction of correct predictions; F1 = 2TP/(2TP+FP+FN) balances precision and recall for the less frequent positive class. Both metrics are needed because an always-negative classifier can appear accurate on an imbalanced dataset. The aim is a transparent LogisticRegression baseline and a controlled comparison after adding two predetermined features; the manual specifies no minimum score. The split, model settings and feature formulas are fixed before measuring either test score. The test set is not used for hyperparameter tuning or feature selection.

The engineering constraints are reproducibility, explicit data-quality reporting and no test-derived fitted statistics. Raw input must be preserved, the split must precede cleaning, and imputation, clipping limits, scaling and category vocabulary must be learned from retained training rows only. Every source row must be accounted for, transformations must be reusable on new data, and a SHA256 fingerprint must identify the exact processed bytes. The intended use is a laboratory study of data engineering. The model does not establish causes of income or suitability for decisions about individuals. The data card records historical bias, sensitive attributes and privacy limitations; the duplicate audit separately identifies dependence that a prescribed random row split may leave across train and test.

## 2. Method

### Ingestion and validation

Used sklearn.datasets.fetch_openml(name='adult', version=2, as_frame=True), OpenML ID 1590. Ingestion logs 48,842 rows and 15 columns without modifying raw values. A canonical raw CSV and the OpenML download cache are stored under the ignored root data/ directory. The original class column is only renamed on a copied view. Raw and test-frame content hashes are checked after processing.

The fixed dictionary in src/schema.py declares all 15 schema types, permitted categories (including ? for feature fields), and the manual's inclusive age 17-90, hours-per-week 1-99, capital-gain 0-99999 and capital-loss 0-4356 ranges. Extra checks cover positive fnlwgt, education-num 1-16 and agreement between education and education-num. Integral numeric values satisfy the integer schema; physical storage dtypes are reported separately. Both parser NaN and literal ? mean feature missingness. Missing target labels and unknown categories are violations. Validation is read-only, reports all failing cells and stops the run on violations; it never silently removes rows. Missing feature values are permitted and counted.

### Split and train-only cleaning

Performed train_test_split with test_size=0.30, stratify=income and random_state=42 before any cleaning or duplicate removal. The split contains {nraw:,} training and {ntest:,} test rows. Exact duplicate FULL rows, including the target, are removed only from training with keep='first', leaving {ntrain:,} rows ({nraw-ntrain} removed). Row IDs and all split assignments are saved. Conflicting-label feature duplicates are audited, never resolved by looking at test labels.

The training-only sklearn Pipeline converts feature ? to missing, fits numeric medians and categorical modes, and learns 1st/99th percentiles on imputed training numeric values. Every original numeric feature is clipped using these fixed limits. Derived features are then computed from those cleaned, clipped values: capital_net = capital-gain - capital-loss; hours_per_age = hours-per-week/(age-16). Validation ensures age>16 and the transformer also guards division. The derived columns are scaled with the original numeric columns; original numeric columns are not removed. There is no separate clipping of derived columns. StandardScaler is fitted on train only. OneHotEncoder learns its vocabulary from train only and uses handle_unknown='ignore'. The test partition keeps every original row and label; only transform is applied to produce model inputs. 'Untouched test' means no deduplication, fitted statistics or label edits, not skipping the required train-fitted transformations.

Both comparisons use LogisticRegression(solver='lbfgs', C=1, max_iter=2000, tol=0.0001, random_state=42) and threshold 0.5. The six numeric features and fnlwgt remain predictors; fnlwgt is not used as sample_weight, so scores are unweighted record-level measures. There is no class weighting or test-driven tuning. Non-convergence is treated as an error. The final engineered transformer is saved at results/preprocessor.joblib; the baseline transformer and both classifiers are also saved.

### Label audit and reproducibility

The full raw dictionary/duplicate audit runs for reporting only and does not change any fitted model. Feature-identical rows with different labels are counted as conflict groups and affected source rows. A fixed seed-42 sample of 50 source rows is saved in audit_sample_50.csv with dictionary checks. The student's manual review worksheet is manual_review_50.csv. It is pending, not represented as completed human evidence, and reruns preserve annotations. A valid dictionary row is not proof that the person's income label is correct. Adult does not require the subjective-label 24-hour relabeling or Cohen's kappa branch of the manual.

Python, NumPy and both prescribed torch seeds use 42. CPU numerical libraries use one thread. Python and every installed distribution are printed and saved. The run exports canonical processed CSV bytes (UTF-8, LF, no index, header, %.17g floats, source IDs and labels included, retained train rows then every test row). Their SHA256 is stored beside the full run config in results/config.yaml. A second fit repeats split, deduplication, preprocessing and both models from the same immutable input; it checks split IDs, arrays, probabilities, predictions and processed SHA256. Joblib reload equivalence and targeted tests are also checked. Hash equality establishes byte identity, not label truth or fairness. Cross-version/hardware bitwise identity is not assumed; the measured environment is recorded.

## 3. Results

{metric_table}

The fixed engineered comparison changes accuracy by {m1['accuracy']-m0['accuracy']:+.6f} and F1 by {m1['f1']-m0['f1']:+.6f}. This is one held-out split, not evidence of statistical significance. Both scores are reported without selecting settings from the test result. Confusion matrices, predictions, convergence iterations, class proportions and clipping counts are available under results/.

![Held-out comparison](results/feature_comparison.png)

Validation found **0 violations** across the specified schema, ranges and additional dictionary checks. Missing counts:

{missing_table}

Raw exact-duplicate excess rows: **{conflicts['exact_duplicate_excess_rows']}**. Feature-duplicate excess rows: **{conflicts['feature_duplicate_excess_rows']}**. Conflicting feature groups: **{conflicts['conflicting_feature_groups']}**; affected rows: **{conflicts['rows_in_conflicting_groups']}**. Exact duplicates removed from training: **{nraw-ntrain}**. Test rows matching a retained training full row: **{overlap['test_rows_matching_train_full_row']}**; matching features regardless of label: **{overlap['test_rows_matching_train_features']}**. The manual's random row split and untouched test policy allow these overlaps. All fitted statistics are isolated from test, but the partitions cannot be claimed fully independent at the entity/duplicate level. A separately specified group split would be needed to study that limitation, and is not substituted for the required split here.

The automated dictionary checks found no impossible values in the fixed 50-row sample. **Student manual review: pending.** Follow manual_review_instructions.md and fill the preserved worksheet. The three conclusions below are draft explanations to review and rewrite in your own words before submission.

Processed dataset SHA256: `{digest}`.

## 4. Three draft conclusions

1. Splitting before cleaning prevents test information from influencing medians, modes, clipping percentiles, scaling or category vocabularies. Persisting the fitted transformer applies exactly those training decisions to later inputs.
2. The two derived features changed held-out F1 by {m1['f1']-m0['f1']:+.6f} on this fixed split. A useful formula does not guarantee an improvement, and this single comparison does not justify tuning against the same test set.
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
"""
    (lab / "report.md").write_text(report,encoding="utf-8")
    validation = f"""# Validation report

Raw ingestion: **48,842 rows x 15 columns**, OpenML Adult v2 / ID 1590.

Result: **PASS - 0 invalid cells**. No validation rows were removed. All 15 expected fields are present; six numeric fields contain integers and nine fields including income satisfy category rules. See schema.json for the complete allowed-category lists and missing_values.csv for per-column storage dtypes.

| Field | Inclusive permitted range | Violations |
|---|---|---:|
| age | 17-90 | 0 |
| hours-per-week | 1-99 | 0 |
| capital-gain | 0-99999 | 0 |
| capital-loss | 0-4356 | 0 |
| education-num (extra) | 1-16 | 0 |
| fnlwgt (extra) | at least 2 | 0 |

Education category/code disagreements: 0. Missing targets: 0. Unknown categories: 0.

{missing_table}

Literal ? is an allowed missing marker; OpenML may already expose it as NaN. These counts are expected imputable missingness, not violations. Imputation starts only after splitting and train-only deduplication.

Train rows: {nraw:,} before deduplication; {ntrain:,} after. Test rows: {ntest:,}; all preserved. Raw full-row duplicate excess: {conflicts['exact_duplicate_excess_rows']}. Conflicting feature groups: {conflicts['conflicting_feature_groups']} ({conflicts['rows_in_conflicting_groups']} source rows). Full-row cross-split matches: {overlap['test_rows_matching_train_full_row']} test rows. No test-driven removal or conflict resolution is performed.

The 50 sampled rows pass automated dictionary checks. The human review remains pending in manual_review_50.csv. Re-run validation failures are written to validation_violations.csv and stop model fitting; previously generated narrative reports should not be treated as current if a later run fails.
"""
    (out / "validation_report.md").write_text(validation,encoding="utf-8")
    card = f"""# Data card - Adult, laboratory version

## Identity and purpose

- Maintainer: {IDENTITY}.
- Educational binary classification and pipeline-quality study; target 1 means income >USD 50,000/year, 0 means <=50K.
- Original: Barry Becker and Ronny Kohavi (1996), Adult, UCI Machine Learning Repository, [DOI 10.24432/C5XW20](https://doi.org/10.24432/C5XW20).
- Sources: [UCI dataset/dictionary](https://archive.ics.uci.edu/dataset/2/adult), [OpenML 1590](https://www.openml.org/d/1590), OpenML adult version 2, default target class. Ingested with the exact prescribed sklearn fetch_openml call.
- Underlying data: extraction from the 1994 US Census database. UCI describes selection AAGE>16, AGI>100, AFNLWGT>1 and HRSWK>0. This is a selected historical sample, not every resident or a present-day population.
- License: the UCI dataset page states [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribute Becker/Kohavi and UCI and disclose transformations. OpenML's metadata uses the less specific string 'Public'; it is preserved in ingestion.json and does not replace the UCI attribution/license statement. Source/license checked 2026-10-27.

## Size, schema and labels

Raw: 48,842 records, 14 predictors + one categorical target. Six integer-valued predictors and eight categorical predictors. No direct personal names or contact details are present. Original row positions serve as reproducibility identifiers, not person IDs. The 70/30 seed-42 stratified split has {nraw:,} raw training / {ntest:,} test rows. Removing {nraw-ntrain} exact full-row duplicate excesses only from training leaves {ntrain:,} training rows and {ntrain+ntest:,} processed rows total. Engineered preprocessing outputs {m1['encoded_features']} numeric/one-hot model columns from 16 pre-encoding features; the baseline has {m0['encoded_features']} outputs from the 14 original features. Counts by label and split are in results/class_counts.csv; category composition by race/sex is in results/composition_*.csv.

The label is a historical income threshold, not a measure of merit, financial health or future income. Label provenance is the source dataset; individual incomes cannot be independently re-verified from these predictors. fnlwgt is a census final weight used as a predictor as the manual supplies all 14 inputs. It is not a unique identifier and is not applied as an evaluation weight.

## Quality and transformations

{missing_table}

Question marks and parser NaN represent missing features. Missingness need not be random; mode imputation may overrepresent majority categories and understate uncertainty. The fixed dictionary/range checks find zero violations. Median numeric imputation, categorical mode imputation, original-numeric 1st/99th percentile clipping, StandardScaler statistics and one-hot vocabulary are fitted only on retained training rows. Clipping is required by the lab and may suppress legitimate extreme income-related values; it is not a correction of proven data errors. The two deterministic formulas capital_net and hours_per_age use cleaned/clipped inputs before scaling. education and education-num encode overlapping information; retaining both follows the full-input baseline.

Raw full-row duplicate excesses: {conflicts['exact_duplicate_excess_rows']}; conflicting feature groups: {conflicts['conflicting_feature_groups']} ({conflicts['rows_in_conflicting_groups']} rows). Conflicting labels are reported without relabeling. These can represent source inconsistencies or different people sharing all recorded attributes. Test rows matching retained train full rows: {overlap['test_rows_matching_train_full_row']}; matching features: {overlap['test_rows_matching_train_features']}. Train-only duplicate removal does not ensure entity-independent holdout results. This limitation follows the manual's row split and untouched-test requirement.

The fixed 50-row sample passes automated dictionary checks; student manual review is pending in results/manual_review_50.csv. The worksheet includes all original values, labels, IDs and fields for findings, reviewer and date. Neither a range pass nor a duplicate audit certifies real-world label accuracy.

## Bias, limitations and use boundaries

Historical selection, class imbalance, nonrandom missingness and unequal representation can distort learned patterns. Race, sex, age, country and relationship are sensitive attributes or plausible proxies; patterns may reflect historical inequalities. The original categories include a binary recorded-sex field and coarse racial groupings that do not represent all identities. US-specific data from 1994 and an unadjusted dollar threshold limit transport across time, locations and populations. Aggregate accuracy/F1 do not establish group fairness, causality, calibration or real deployment utility. No causal or population-representative claim is made, and no fairness mitigation or subgroup performance certification is performed. Use as a teaching benchmark, not as sole evidence for decisions about employment, lending, benefits or individuals.

## Privacy and storage

Removing direct identifiers does not eliminate re-identification risk: combinations of demographic and work attributes are quasi-identifiers, and income is sensitive. Do not link these rows to identified people. SHA256 is an integrity fingerprint, not encryption or anonymization. Raw data, OpenML cache and processed matrices stay under ignored root data/. Reports contain aggregate counts; the manual-review and conflict evidence includes public source records. No private student information is added to dataset rows. Joblib artifacts are trusted local generated files and depend on importable src/transforms.py and matching library versions.

## Reproducibility and distribution

Run python lab04/main.py from the course root using Python 3.11.8 and requirements-lab04.txt. The first run downloads OpenML data; subsequent runs use the sklearn cache. Source, config, split IDs, training statistics, fitted preprocessors/classifiers, predictions, exact installed versions and verification evidence are retained. Original course requirements.txt stays unchanged; its pre-existing mlflow/pyarrow conflict is documented in the earlier labs, so the compatible lab subset is used here.

Processed file: data/lab04/processed_dataset.csv. UTF-8/LF CSV, header, no index, %.17g float formatting, retained train then all test rows, split + source row_id + binary label + ordered model feature names. SHA256: **{digest}**. results/config.yaml contains this hash and the full run configuration. results/reproducibility.json checks an independent second fit and reload identity. Byte identity is verified in the recorded environment, not guaranteed across arbitrary software/hardware changes. The submission archive excludes root data/; the runnable code recreates it from the source. Preserve the downloaded snapshot/cache when exact offline reruns are needed.

## Review and maintenance

Version: Lab 4 / 2026-10-27. Refresh validation, class/missingness counts, hashes and documentation whenever source, split, schema or transformations change. Preserve completed manual annotations; obtain a fresh review if sampled IDs change. Review the AI-assisted code and draft conclusions before personal submission. No claim of completed human review is made.
"""
    (lab / "data_card.md").write_text(card,encoding="utf-8")
