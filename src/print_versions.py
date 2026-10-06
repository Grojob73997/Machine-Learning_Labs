"""Print the Python and library versions required by the laboratory manual."""

from __future__ import annotations

from importlib import metadata
import platform
import sys


PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "torch": "torch",
    "torchvision": "torchvision",
    "torchinfo": "torchinfo",
    "thop": "thop",
    "onnx": "onnx",
    "onnxruntime": "onnxruntime",
    "mlflow": "mlflow",
    "memory-profiler": "memory_profiler",
    "psutil": "psutil",
    "codecarbon": "codecarbon",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pytest": "pytest",
    "httpx": "httpx",
    "locust": "locust",
    "requests": "requests",
    "pyarrow": "pyarrow",
    "joblib": "joblib",
    "tqdm": "tqdm",
}


def main() -> None:
    print(f"Python: {platform.python_version()}")
    print(f"Python implementation: {platform.python_implementation()}")
    print(f"Platform: {platform.platform()}")
    for requirement_name, import_name in PACKAGES.items():
        try:
            version = metadata.version(requirement_name)
            print(f"{requirement_name}: {version}")
        except metadata.PackageNotFoundError:
            print(f"{requirement_name}: NOT INSTALLED")


if __name__ == "__main__":
    main()
