# G M Robi Farhan Rojob | 73997 | Lab 2
# Run this file to execute tasks 1 through 7.
# Earlier tasks provide the data and functions used below.
from pathlib import Path
from runpy import run_path

previous_tasks = run_path(str(Path(__file__).with_name('task-6.py')))
globals().update({name: value for name, value in previous_tasks.items()
                  if not name.startswith('__')})

# TASK 7 - FORWARD/BACKWARD TIMING
print("\nTASK 7")
timing_rows = []
for (width, rate), p in trained_models.items():
    X_batch, Y_batch = X_train[:32], Y_train[:32]
    _, cache = forward(X_batch, p)
    forward(X_batch, p)
    backward(p, cache, Y_batch)
    forward_times, backward_times = [], []
    for _ in range(100):
        start = time.perf_counter_ns()
        forward(X_batch, p)
        forward_times.append(time.perf_counter_ns() - start)
        start = time.perf_counter_ns()
        backward(p, cache, Y_batch)
        backward_times.append(time.perf_counter_ns() - start)
    forward_us = float(np.median(forward_times) / 1000)
    backward_us = float(np.median(backward_times) / 1000)
    timing_rows.append(dict(width=width, learning_rate=rate,
                            forward_median_us=forward_us, backward_median_us=backward_us))
    print(f"W={width} lr={rate}: forward={forward_us:.2f} us, backward={backward_us:.2f} us")
save_csv("today_timing.csv", timing_rows)
print("\nCODE COMPLETE. Results folder:", RESULTS)

