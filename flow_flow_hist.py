#!/usr/bin/env python3
"""
Aggregate flow values from the tag directories and build a histogram +
40% percentile report.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np


def iter_json_files(dir_path: Path) -> Iterable[Path]:
    if not dir_path.exists():
        print(f"[WARN] Directory not found: {dir_path}", file=sys.stderr)
        return []
    if not dir_path.is_dir():
        print(f"[WARN] Not a directory, skipped: {dir_path}", file=sys.stderr)
        return []
    return sorted(p for p in dir_path.glob("*.json") if p.is_file())


def collect_flows(directories: List[Path]) -> List[float]:
    flows: List[float] = []
    for directory in directories:
        for json_path in iter_json_files(directory):
            try:
                with json_path.open("r", encoding="utf-8") as fp:
                    payload = json.load(fp)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[WARN] Failed to load {json_path}: {exc}", file=sys.stderr)
                continue

            if isinstance(payload, list):
                entries = payload
            elif isinstance(payload, dict):
                entries = payload.values()
            else:
                print(f"[WARN] Unrecognized structure in {json_path}, skipped", file=sys.stderr)
                continue

            for item in entries:
                if not isinstance(item, dict):
                    continue
                value = item.get("flow")
                if isinstance(value, (int, float)):
                    flows.append(float(value))
                elif isinstance(value, list):
                    flows.extend(float(v) for v in value if isinstance(v, (int, float)))
                else:
                    continue
    if not flows:
        raise RuntimeError("No flow data found in the provided directories.")
    return flows


def plot_hist(flows: List[float], output_path: Path, bins: int = 80) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(flows, bins=bins, color="#1f77b4", edgecolor="black", alpha=0.75)
    plt.xlabel("flow")
    plt.ylabel("Count")
    plt.title("Flow histogram (data_2003_tag + data_269_tag)")
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"[INFO] Histogram saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    default_dirs = [
        Path("/share/project/denghaoge/shilinlu/dataset/GF-Minecraft/data_2003_tag"),
        Path("/share/project/denghaoge/shilinlu/dataset/GF-Minecraft/data_269_tag"),
    ]

    parser = argparse.ArgumentParser(description="Compute flow histogram and 40% percentile")
    parser.add_argument(
        "--dirs",
        nargs="+",
        type=Path,
        default=default_dirs,
        help="Directories to scan (default: data_2003_tag and data_269_tag)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("flow_hist.png"),
        help="Histogram output path (default: flow_hist.png)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=80,
        help="Histogram bin count (default: 80)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directories = [d.resolve() for d in args.dirs]
    flows = collect_flows(directories)
    flows_arr = np.array(flows, dtype=np.float64)
    acc40 = np.percentile(flows_arr, 40)
    print(f"[RESULT] Flow value at cumulative 40%% (acc=40%%): {acc40:.6f}")
    plot_hist(flows, args.output.resolve(), bins=args.bins)


if __name__ == "__main__":
    main()

