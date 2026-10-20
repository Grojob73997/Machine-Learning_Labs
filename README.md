# Machine Learning Systems Laboratory Repository

This repository contains the reproducible work for the Machine Learning Systems laboratory sequence.

## Lab 1

Lab 1 creates a pinned Python environment, trains Logistic Regression and Random Forest baselines on the Breast Cancer Wisconsin diagnostic dataset, measures system cost, and evaluates deployment against Cloud, Edge, Mobile, and TinyML budgets.

Run the lab from the repository root with:

```text
python lab01/src/print_versions.py > lab01/results/versions.txt
python lab01/src/run_lab01.py
```

The required Python version is 3.11.8. Install the pinned dependencies with `pip install -r requirements.txt` inside an isolated environment.

## Lab 3

Lab 3 compares the exact MLP, CNN and attention architectures on Fashion-MNIST.
Run `python lab03/main.py` using Python 3.11.8 and the dependencies in
`lab03/requirements-lab03.txt`. The same tested interpreter used for Lab 2 now
also contains the required Lab 3 packages. See `lab03/RUN_IN_PYCHARM.txt`.

The report is `lab03/report.md`, measured evidence is in `lab03/results/`,
and the personal guide is `lab03/lab3 explanation.pdf`. Question 3 still
requires the student's handwritten calculations: follow `hand_calculations.md`
and add scans under `lab03/results/handwritten/`.

The exact architecture specifications conflict with the 100k-150k parameter
target. The report keeps the specified layers and explains the discrepancy.
It uses the supplied July 31, 2025 textbook edition and documents profiler
coverage and the existing course-wide package dependency conflict.
