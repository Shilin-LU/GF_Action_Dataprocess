#!/usr/bin/env python3
"""
Add a `flow` field to each entry in the JSON annotation files by averaging
the existing `flows` values.

This script walks through the GF-Minecraft tag directories and updates every
`*.json` file it finds so that each dictionary element gains a `flow` key whose
value is the arithmetic mean of its `flows` list.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence


DEFAULT_DIRECTORIES: Sequence[Path] = (
    Path("/share/project/denghaoge/shilinlu/dataset/GF-Minecraft/data_2003_tag"),
    Path("/share/project/denghaoge/shilinlu/dataset/GF-Minecraft/data_269_tag"),
)


def discover_json_files(directories: Sequence[Path]) -> Iterable[Path]:
    """Yield JSON files under each directory in sorted order."""
    for directory in directories:
        if not directory.exists():
            continue
        yield from sorted(directory.glob("*.json"))


def compute_average(flows: Sequence[float]) -> float | None:
    """Return the arithmetic mean for a non-empty sequence of numbers."""
    if not flows:
        return None
    total = 0.0
    count = 0
    for value in flows:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        total += numeric
        count += 1
    if count == 0:
        return None
    return total / count


def update_file(path: Path, dry_run: bool) -> int:
    """Update a JSON file and return the number of entries modified."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        return 0

    modifications = 0
    for entry in data:
        if not isinstance(entry, dict):
            continue
        flows = entry.get("flows")
        if not isinstance(flows, list):
            continue

        average = compute_average(flows)
        if average is None:
            continue

        if entry.get("flow") != average:
            entry["flow"] = average
            modifications += 1

    if modifications and not dry_run:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False)

    return modifications


def run(directories: Sequence[Path], dry_run: bool) -> None:
    total_files = 0
    total_entries = 0

    for json_file in discover_json_files(directories):
        modified = update_file(json_file, dry_run=dry_run)
        if modified:
            total_files += 1
            total_entries += modified
            status = "would update" if dry_run else "updated"
            print(f"{status} {json_file} ({modified} entries)")

    summary_action = "Previewed" if dry_run else "Updated"
    print(f"{summary_action} {total_entries} entries across {total_files} files.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add flow averages to GF-Minecraft tag JSON files."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Specific directories to process (defaults to known tag directories).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes without writing back to disk.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    directories: List[Path] = (
        list(args.paths) if args.paths else list(DEFAULT_DIRECTORIES)
    )
    run(directories, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

