# Data card - Adult, laboratory version

## Identity and purpose

- Maintainer: G M Robi Farhan Rojob | Student number 73997 | CII5SE039CI | Group 01.
- Educational binary classification and pipeline-quality study; target 1 means income >USD 50,000/year, 0 means <=50K.
- Original: Barry Becker and Ronny Kohavi (1996), Adult, UCI Machine Learning Repository, [DOI 10.24432/C5XW20](https://doi.org/10.24432/C5XW20).
- Sources: [UCI dataset/dictionary](https://archive.ics.uci.edu/dataset/2/adult), [OpenML 1590](https://www.openml.org/d/1590), OpenML adult version 2, default target class. Ingested with the exact prescribed sklearn fetch_openml call.
- Underlying data: extraction from the 1994 US Census database. UCI describes selection AAGE>16, AGI>100, AFNLWGT>1 and HRSWK>0. This is a selected historical sample, not every resident or a present-day population.
- License: the UCI dataset page states [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribute Becker/Kohavi and UCI and disclose transformations. OpenML's metadata uses the less specific string 'Public'; it is preserved in ingestion.json and does not replace the UCI attribution/license statement. Source/license checked 2026-10-27.

## Size, schema and labels

Raw: 48,842 records, 14 predictors + one categorical target. Six integer-valued predictors and eight categorical predictors. No direct personal names or contact details are present. Original row positions serve as reproducibility identifiers, not person IDs. The 70/30 seed-42 stratified split has 34,189 raw training / 14,653 test rows. Removing 24 exact full-row duplicate excesses only from training leaves 34,165 training rows and 48,818 processed rows total. Engineered preprocessing outputs 107 numeric/one-hot model columns from 16 pre-encoding features; the baseline has 105 outputs from the 14 original features. Counts by label and split are in results/class_counts.csv; category composition by race/sex is in results/composition_*.csv.

The label is a historical income threshold, not a measure of merit, financial health or future income. Label provenance is the source dataset; individual incomes cannot be independently re-verified from these predictors. fnlwgt is a census final weight used as a predictor as the manual supplies all 14 inputs. It is not a unique identifier and is not applied as an evaluation weight.

## Quality and transformations

| Column | Parser nulls | Literal ? | Total missing |
|---|---|---|---|
| workclass | 2799 | 0 | 2799 |
| occupation | 2809 | 0 | 2809 |
| native-country | 857 | 0 | 857 |

Question marks and parser NaN represent missing features. Missingness need not be random; mode imputation may overrepresent majority categories and understate uncertainty. The fixed dictionary/range checks find zero violations. Median numeric imputation, categorical mode imputation, original-numeric 1st/99th percentile clipping, StandardScaler statistics and one-hot vocabulary are fitted only on retained training rows. Clipping is required by the lab and may suppress legitimate extreme income-related values; it is not a correction of proven data errors. The two deterministic formulas capital_net and hours_per_age use cleaned/clipped inputs before scaling. education and education-num encode overlapping information; retaining both follows the full-input baseline.

Raw full-row duplicate excesses: 52; conflicting feature groups: 5 (10 rows). Conflicting labels are reported without relabeling. These can represent source inconsistencies or different people sharing all recorded attributes. Test rows matching retained train full rows: 24; matching features: 27. Train-only duplicate removal does not ensure entity-independent holdout results. This limitation follows the manual's row split and untouched-test requirement.

The fixed 50-row sample passes automated dictionary checks; student manual review is pending in results/manual_review_50.csv. The worksheet includes all original values, labels, IDs and fields for findings, reviewer and date. Neither a range pass nor a duplicate audit certifies real-world label accuracy.

## Bias, limitations and use boundaries

Historical selection, class imbalance, nonrandom missingness and unequal representation can distort learned patterns. Race, sex, age, country and relationship are sensitive attributes or plausible proxies; patterns may reflect historical inequalities. The original categories include a binary recorded-sex field and coarse racial groupings that do not represent all identities. US-specific data from 1994 and an unadjusted dollar threshold limit transport across time, locations and populations. Aggregate accuracy/F1 do not establish group fairness, causality, calibration or real deployment utility. No causal or population-representative claim is made, and no fairness mitigation or subgroup performance certification is performed. Use as a teaching benchmark, not as sole evidence for decisions about employment, lending, benefits or individuals.

## Privacy and storage

Removing direct identifiers does not eliminate re-identification risk: combinations of demographic and work attributes are quasi-identifiers, and income is sensitive. Do not link these rows to identified people. SHA256 is an integrity fingerprint, not encryption or anonymization. Raw data, OpenML cache and processed matrices stay under ignored root data/. Reports contain aggregate counts; the manual-review and conflict evidence includes public source records. No private student information is added to dataset rows. Joblib artifacts are trusted local generated files and depend on importable src/transforms.py and matching library versions.

## Reproducibility and distribution

Run python lab04/main.py from the course root using Python 3.11.8 and requirements-lab04.txt. The first run downloads OpenML data; subsequent runs use the sklearn cache. Source, config, split IDs, training statistics, fitted preprocessors/classifiers, predictions, exact installed versions and verification evidence are retained. Original course requirements.txt stays unchanged; its pre-existing mlflow/pyarrow conflict is documented in the earlier labs, so the compatible lab subset is used here.

Processed file: data/lab04/processed_dataset.csv. UTF-8/LF CSV, header, no index, %.17g float formatting, retained train then all test rows, split + source row_id + binary label + ordered model feature names. SHA256: **50a22304e1125a60d773b504cc25c762597e74732c244b8bd8abfbce317cd115**. results/config.yaml contains this hash and the full run configuration. results/reproducibility.json checks an independent second fit and reload identity. Byte identity is verified in the recorded environment, not guaranteed across arbitrary software/hardware changes. The submission archive excludes root data/; the runnable code recreates it from the source. Preserve the downloaded snapshot/cache when exact offline reruns are needed.

## Review and maintenance

Version: Lab 4 / 2026-10-27. Refresh validation, class/missingness counts, hashes and documentation whenever source, split, schema or transformations change. Preserve completed manual annotations; obtain a fresh review if sampled IDs change. Review the AI-assisted code and draft conclusions before personal submission. No claim of completed human review is made.
