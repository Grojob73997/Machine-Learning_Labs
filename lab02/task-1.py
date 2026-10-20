"""Compatibility entry point. The task implementations are in src/."""
from pathlib import Path
from runpy import run_path
run_path(str(Path(__file__).resolve().parent / "src" / "task-1.py"), run_name="__main__")
