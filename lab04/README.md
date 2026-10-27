# Lab 4 - Adult data pipeline and quality

G M Robi Farhan Rojob | 73997 | Machine Learning, Group 01

Run **lab04/main.py** in the same course project as Labs 1-3. The root main.py is the old Lab 2 launcher.

```powershell
python lab04/main.py
```

Use Python 3.11.8 with `lab04/requirements-lab04.txt`. The tested existing interpreter is:

`C:\Users\rojob\Documents\ChatGPT\Machine Learning\.venv-lab2\Scripts\python.exe`

It now includes pandas 2.2.2 and PyYAML 6.0.2 as well as the earlier lab packages. To set up a different machine, create a virtual environment and install the Lab 4 requirements. The root course package list is preserved; its earlier documented mlflow/pyarrow conflict remains, so it is not the install command for this lab.

The one command ingests OpenML Adult version 2, validates raw data, stratifies 70/30, removes exact train duplicates, fits train-only cleaning/encoding, compares both logistic models, audits labels, exports the processed hash/config and data card, repeats both fits to verify reproducibility, and runs unit plus fresh-process artifact checks. It recreates ignored `data/` outputs when needed. First use requires OpenML access; the sklearn cache supports later offline runs. All output paths resolve from the source location, so PyCharm working-directory choices do not change them.

Read `report.md`, `data_card.md`, `results/validation_report.md` and the companion explanation PDF. `TASK_CHECKLIST.md` maps each manual task to evidence. `results/preprocessor.joblib` is the fitted engineered transformer; its custom classes live in `src/transforms.py`. To reload it from your own script, put `lab04/src` on the Python import path first. `src/verify_lab.py` demonstrates a fresh-process reload and checks the current saved artifacts without retraining:

```powershell
python lab04/src/verify_lab.py
```

The raw input and processed matrices are stored under root `data/lab04/`; the OpenML cache is `data/openml/`. Data is ignored by Git and excluded from the submission archive. The SHA256 refers to the exact canonical processed CSV, including both partitions, binary labels and source IDs. See the serialization contract in `results/config.yaml`.

**Before personal submission:** complete your own 50-row review in `results/manual_review_50.csv`, review/rewrite the three draft conclusions, understand the implementation, and commit those changes. The fixed worksheet is preserved across reruns. Automated dictionary checks are completed; human inspection is not claimed. Full-row duplicate overlap remains across partitions under the required random row split and is reported as an evaluation limitation.

This is AI-assisted source and draft documentation. The manual requires individual work and conclusions in your own words. The package supplies measured evidence and a study guide; it does not certify your attendance, human review or personal understanding.
