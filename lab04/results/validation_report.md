# Validation report

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

| Column | Parser nulls | Literal ? | Total missing |
|---|---|---|---|
| workclass | 2799 | 0 | 2799 |
| occupation | 2809 | 0 | 2809 |
| native-country | 857 | 0 | 857 |

Literal ? is an allowed missing marker; OpenML may already expose it as NaN. These counts are expected imputable missingness, not violations. Imputation starts only after splitting and train-only deduplication.

Train rows: 34,189 before deduplication; 34,165 after. Test rows: 14,653; all preserved. Raw full-row duplicate excess: 52. Conflicting feature groups: 5 (10 source rows). Full-row cross-split matches: 24 test rows. No test-driven removal or conflict resolution is performed.

The 50 sampled rows pass automated dictionary checks. The human review remains pending in manual_review_50.csv. Re-run validation failures are written to validation_violations.csv and stop model fitting; previously generated narrative reports should not be treated as current if a later run fails.
