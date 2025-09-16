import argparse
import json
import os
from pathlib import Path

import models.llm as llm


REPO_ROOT = Path(__file__).parent.parent
OLD_RESULTS_DIR = REPO_ROOT / "results" / "llm" / "old"


def detect_latest_results_files() -> list[Path]:
    patterns = [
        "-home-derposoft-code-plot_hole_detection_next-data-dataset-1_error-test_*.json",
        "-home-derposoft-code-plot_hole_detection_next-data-dataset-2_error-test_*.json",
        "-home-derposoft-code-plot_hole_detection_next-data-dataset-5_error-test_*.json",
    ]

    found: list[Path] = []
    for pattern in patterns:
        matches = sorted(OLD_RESULTS_DIR.glob(pattern), key=os.path.getmtime)
        if matches:
            found.append(matches[-1])
    return found


def load_results(path: Path) -> dict[str, list[int]]:
    with open(path, "r") as f:
        return json.load(f)


def print_file_header(path: Path):
    name = path.name
    if "1_error" in name:
        print("1-error metrics:")
    elif "2_error" in name:
        print("2-error metrics:")
    elif "5_error" in name:
        print("5-error metrics:")
    else:
        print(f"Metrics for {name}:")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-file",
        action="append",
        help="Path to a results JSON produced by llm_fix1 or llm.py",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    files: list[Path]
    if args.results_file:
        files = [Path(p) for p in args.results_file]
    else:
        files = detect_latest_results_files()

    if not files:
        print("No results files found.")
        return

    for path in files:
        try:
            results = load_results(path)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            continue
        print_file_header(path)
        llm.print_metrics(results)


if __name__ == "__main__":
    main()
