# Lab 4 evidence checklist

| Manual task | Evidence | Status |
|---|---|---|
| 1. Half-page problem definition, 14 inputs, >50K, accuracy/F1, constraints | report.md section 1 | Prepared; student review required |
| 2. Immutable raw ingestion, 48,842 x 15 | results/ingestion.json; data/lab04/adult_openml_v2_raw.csv | Executed |
| 3. Schema, ranges, ? policy, violations | src/schema.py; results/schema.json; validation_report.md; validation_violations.csv; missing_values.csv | Executed |
| 4. Stratified 70/30 before cleaning; train imputation/clipping; train-only dedup; joblib | src/run_lab.py; src/transforms.py; split_indices.csv; fitted_parameters.json; preprocessor.joblib | Executed |
| 5. Both derived features and logistic comparison | metrics.csv; predictions_*.csv; feature_comparison.png | Executed |
| 6. 50 random rows and duplicate-conflict audit | audit_sample_50.csv; quality_audit.json; conflicting_labels.csv | Automated portion executed |
| 6. Manual dictionary inspection | manual_review_50.csv; manual_review_instructions.md | Pending student review |
| 7. SHA256 + YAML + complete data card + one command | processed_dataset.sha256; results/config.yaml; data_card.md; main.py | Executed |
| Global reproducibility/version evidence | versions.txt; requirements-lock.txt; reproducibility.json; verification.txt | Executed |
| Global report with three own-word conclusions | report.md section 4 | Draft; personalize |
| Textbook preparation | study_guide.md and explanation PDF; reference_record.json | Guide supplied; student reading remains |
| Bring a digital/printed manual | references/Lab Manual_ Labs 04.pdf | Digital copy supplied |

Paths in the evidence column that omit a directory refer to `lab04/results/`, except named source/docs at the lab root. Root `data/` is ignored and recreated by the pipeline. No completed personal review or handwriting is fabricated.
