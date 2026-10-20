"""PyCharm entry point: right-click this file and choose Run."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from run_lab import main

if __name__ == "__main__":
    main()
