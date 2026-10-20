"""Run this file in PyCharm to open the Lab 2 interface in your browser."""
from pathlib import Path
import webbrowser

page = Path(__file__).resolve().parent / "ui" / "index.html"
if not (page.parent / "data.js").exists():
    raise SystemExit("Run src/build_ui.py once to create the experiment data.")
webbrowser.open(page.as_uri())
print("Opened:", page)

