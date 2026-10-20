"""Print Python and every installed library version; compare all course pins."""
from importlib import metadata
from pathlib import Path
import platform


def version_report():
    root = Path(__file__).resolve().parents[2]
    lines = [f"Python: {platform.python_version()}", f"Platform: {platform.platform()}",
             "Required Python: 3.11.8", f"Python match: {platform.python_version() == '3.11.8'}",
             "", "COURSE REQUIREMENTS"]
    for item in (root / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if "==" not in item or item.startswith("#"):
            continue
        name, expected = item.split("==")
        try:
            installed = metadata.version(name)
            status = "MATCH" if installed == expected else "DIFFERENT"
        except metadata.PackageNotFoundError:
            installed, status = "not installed (not used in Lab 2)", "ABSENT"
        lines.append(f"{name}: expected {expected}; installed {installed}; {status}")
    lines += ["", "ALL INSTALLED DISTRIBUTIONS"]
    lines.extend(sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    output = version_report()
    print(output)
    path = Path(__file__).resolve().parents[1] / "results" / "versions.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(output, encoding="utf-8")
