# Every instruction in the Lab 3 manual

The bracketed values [10], [25], [25], [30], [15], [10] are estimated minutes,
not marks: the global rules say so. They sum to 115 minutes.

| Requirement | Evidence / action |
|---|---|
| Read Reddi 4.2, 4.3, 4.5, 4.6, 4.7 | Exact supplied July 2025 edition checked; page map in report.md. Student should read these sections too. |
| 1: torchvision FashionMNIST; ToTensor; exact Normalize | src/run_lab.py, load_data |
| 1: split official 60k into 55k/5k; official 10k test | results/split_indices.npz; class_counts.csv; data hashes |
| 1: batch 128 | All loaders; 430 train updates/epoch; final train batch 88 |
| 2: exact MLP | src/models.py; torchinfo_MLP.txt |
| 2: exact CNN and padding/pooling | src/models.py; torchinfo_CNN.txt |
| 2: exact patch attention, two encoders, heads, feedforward, pooling | src/models.py; torchinfo_Attention.txt |
| 2: target 100k-150k parameters | Conflicts with all explicit architectures; discrepancy disclosed, not falsely marked satisfied |
| 3: torchinfo.summary parameter count | Three complete summary files; cross-check against direct enumeration |
| 3: hand calculation for linear, convolution, attention | hand_calculations.md and separate study PDF contain full working; student must handwrite on paper |
| 3: compare with thop.profile; explain gap | compute_costs.csv; native logs; aligned custom-rule counts; report explanation |
| 3: batch-1 median of 100; batch-128 median of 20 | latency_raw.csv: 360 records; summary.csv |
| 4: all three models; 5 epochs; Adam; lr=1e-3; batch 128 | epoch_log.csv: 15 rows; model checkpoints; run_log.txt |
| 4: record test accuracy | summary.csv; verified by reloading all three checkpoints |
| 5: accuracy vs parameters | accuracy_vs_parameters.png |
| 5: accuracy vs latency | accuracy_vs_latency_batch1.png and accuracy_vs_latency_batch128.png |
| 5: explain cost effectiveness using 4.6-4.7 access patterns | report.md discussion with exact book references and limitations |
| 6: report.md | Goal, method, results, three conclusions, references; no code listings |
| 6: commit | See DELIVERY_STATUS.txt for actual commit outcome |
| Global: same student course folder, lab03/src and lab03/results | Added to existing E-drive course folder alongside lab01 and lab02 |
| Global: data at repository root and ignored | data/FashionMNIST; existing root .gitignore has data/ |
| Global: four seed calls with 42 | src/run_lab.py, seed_all |
| Global: warm-up, repeats, medians | One warm-up for each batch/model; exact Lab 3 repeat counts |
| Global: Python 3.11.8 and print all versions | results/versions.txt and environment.json |
| Global: exact pinned course stack | Root pins preserved; actual Lab 3 subset matches pins. Full course stack has documented mlflow/pyarrow conflict |
| Global: individual work / conclusions in own words | Student must understand the work, complete the paper calculations, review the report and rewrite conclusions in their own words |

No marks or accuracy thresholds are invented. No approval of the contradictory
parameter target is claimed. The extra whole-model arithmetic, batch-128 plot,
checkpoints and verification make the required evidence easier to inspect.
