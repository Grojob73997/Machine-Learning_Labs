# Student review of the 50 sampled rows

The manual asks you to **manually** re-check 50 rows against the data dictionary. Automated checks do not establish that you performed that review. This package therefore leaves the human fields blank.

1. Open `results/manual_review_50.csv` in a spreadsheet or text editor. Its seed-42 source IDs and original values match `results/audit_sample_50.csv`.
2. For each row, compare all 14 predictors and the income label with the [UCI dictionary](https://archive.ics.uci.edu/dataset/2/adult) and the local lists in `results/schema.json`.
3. Check integer values, age 17-90, hours 1-99, gain 0-99999, loss 0-4356, education number 1-16, positive census weight, recognized categories, and education/category agreement. `?` is permitted missingness in feature fields; it is not automatically an impossible value. The target must be <=50K or >50K.
4. Look at combinations as well as single fields. Note suspicious combinations for review without asserting they are impossible: e.g. young age/advanced education can be unusual but a range check alone cannot determine the true situation. `fnlwgt` is a sampling weight, not salary. `capital-gain` is not the income target.
5. In `manual_result`, enter PASS, FLAG or UNCERTAIN after your review. State the rule or concern in `manual_notes`; add your real name and actual review date in the remaining fields. Do not change source values or the source row IDs. A pass means no dictionary inconsistency found, not verified real-world income.
6. Read `results/conflicting_labels.csv`: it contains ten records from five feature-identical groups with different labels. These cannot be automatically resolved from the available predictors. No target is overwritten.
7. When all 50 reviews are complete, add a short, accurate note to the report, review the three draft conclusions and commit your changes. If you flag an issue, document it; do not silently delete rows or refit using test-derived choices.

`main.py` preserves this worksheet on reruns and rejects mismatched sample IDs. It still conservatively describes human review as pending; update your report/status after you have actually completed it. Adult uses the duplicate-conflict audit branch: the manual's wait-24-hours/relabel/kappa procedure is for subjective-label datasets, not this task.
