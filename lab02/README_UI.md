# Lab 2 — Python tasks and browser interface

G M Robi Farhan Rojob · 73997 · Group 01

## Open the interface

Run `open_ui.py` in PyCharm, or open `ui/index.html` in your browser.
No server, internet connection, Node installation, or extra Python packages are needed to view it.

The seven navigation items correspond to tasks 1–7 in the manual. Change hidden width
and learning rate to inspect the six completed experiments. Task 3 lets you select test
digits and see predictions from the exported trained weights. These controls select
existing experiments; they do not train new models in the browser.

## Run or change the Python work

Use Python 3.11.8 with the packages in `requirements-lab02.txt`.
The prepared local interpreter is:

`C:\Users\rojob\Documents\ChatGPT\Machine Learning\.venv-lab2\Scripts\python.exe`

This interpreter is outside this project folder. If moving the project to another computer,
create a Python 3.11.8 environment and install `requirements-lab02.txt` there.

- `src/task-1.py`: dataset, scaling, splitting, one-hot labels.
- `src/task-2.py`: initialization, parameter shapes, Python network diagrams.
- `src/task-3.py`: forward pass and assertions.
- `src/task-4.py`: cross-entropy and baseline check.
- `src/task-5.py`: backpropagation and numerical gradient checking.
- `src/task-6.py`: six training experiments and loss curves.
- `src/task-7.py`: separate forward/backward timing.

Each task automatically executes its preceding tasks to obtain the necessary variables.
Running task 7 therefore executes all seven tasks. Keep the files together in `src`.

After editing Python code, run `src/build_ui.py` to rerun the experiments and regenerate
`ui/data.js`. Refresh the browser to see the new measurements. Running a task alone updates
its CSV/PNG outputs but does not update the browser's exported data until the builder runs.

## Implementation and evidence

`ui/index.html`, `styles.css`, and `app.js` define the interface. `network.js` implements
only forward inference in JavaScript; training and numerical checks remain in Python.
`data.js` contains actual exported parameters, dataset samples, and measurements.
All graphics use local SVG, canvas, CSS, or Python/Matplotlib; there are no external services.

The numerical reference loss uses 50-digit Decimal arithmetic for the gradient check;
training uses NumPy float64. Gaussian initialization uses standard deviation 0.01.
The low-accuracy experiments are reported as measured, without adjusting the prescribed setup.

The teacher-approved Python architecture drawing is in `results/task-2-network-diagram.pdf`.
The original manual's hand-derived gradient requirement remains separate from the interface.
