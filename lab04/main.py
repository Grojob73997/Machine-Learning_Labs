"""Run Lab 4 from PyCharm or with: python lab04/main.py."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from run_lab import main

if __name__ == "__main__":
    main()
