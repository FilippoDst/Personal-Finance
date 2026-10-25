"""Runs the whole pipeline in order (from the repository root): python run_all.py"""
import subprocess
import sys
from pathlib import Path

for script in sorted(Path("src").glob("0*.py")):
    print(f"\n=== {script.name} ===", flush=True)
    result = subprocess.run([sys.executable, str(script)])
    if result.returncode != 0:
        sys.exit(result.returncode)
