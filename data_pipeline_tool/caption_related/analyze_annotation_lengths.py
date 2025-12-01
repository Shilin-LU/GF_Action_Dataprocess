#!/usr/bin/env python3
"""
Analyze the prompt word-count distribution of annotation CSV files.

Typical usage:
    python analyze_annotation_lengths.py \
        --files data_2003/annotation_action.csv data_269/annotation_action.csv \
        --bins 60 \
        --output annotation_prompt_word_lengths.png \
        --target-words 500 \
        --tolerance 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot prompt word-count histograms for annotation CSV files."
    )
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="Absolute or relative paths to annotation CSV files.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of bins to use for the histograms (default: 50).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the histogram figure (PNG). If omitted, the figure is shown interactively.",
    )
    parser.add_argument(
        "--target-words",
        type=int,
        default=None,
        help="Optional target word count to search for prompts around.",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=10,
        help="Allowed difference (in words) when searching for prompts near the target word count (default: 10).",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=5,
        help="Maximum number of prompts to display when target word count search is enabled (default: 5).",
    )
    return parser.parse_args(argv)


def load_prompt_lengths(csv_path: Path) -> Tuple[pd.Series, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    if "prompt" not in df.columns:
        raise ValueError(f"Expected a 'prompt' column in {csv_path}")
    prompts = df["prompt"].fillna("").astype(str)
    word_counts = prompts.str.split().apply(len)
    return word_counts, df


def summarize_lengths(label: str, lengths: pd.Series) -> None:
    print(f"=== {label} ===")
    print(f"count: {len(lengths)}")
    print(f"min:   {lengths.min()}")
    print(f"p25:   {lengths.quantile(0.25):.2f}")
    print(f"median:{lengths.median():.2f}")
    print(f"p75:   {lengths.quantile(0.75):.2f}")
    print(f"max:   {lengths.max()}")
    print(f"mean:  {lengths.mean():.2f}")
    print(f"std:   {lengths.std():.2f}")
    print()


def plot_histograms(
    lengths_by_label: List[Tuple[str, pd.Series]], bins: int, output: Path | None
) -> None:
    num_files = len(lengths_by_label)
    if num_files == 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        axes = [ax]
    else:
        cols = min(2, num_files)
        rows = (num_files + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 4 * rows), squeeze=False)
        axes = axes.ravel()

    for ax, (label, lengths) in zip(axes, lengths_by_label):
        ax.hist(lengths, bins=bins, edgecolor="black", alpha=0.7)
        ax.set_title(label)
        ax.set_xlabel("Prompt word count")
        ax.set_ylabel("Frequency")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Hide any unused axes if we created more subplots than needed.
    for ax in axes[len(lengths_by_label):]:
        ax.remove()

    fig.tight_layout()

    if output:
        output = output.with_suffix(".png")
        fig.savefig(output, dpi=200, bbox_inches="tight")
        print(f"Histogram saved to {output}")
    else:
        plt.show()


def analyze(
    files: Iterable[Path],
    bins: int,
    output: Path | None,
    target_words: int | None,
    tolerance: int,
    max_prompts: int,
) -> None:
    lengths_by_label: List[Tuple[str, pd.Series]] = []

    for path in files:
        lengths, df = load_prompt_lengths(path)
        summarize_lengths(path.name, lengths)

        p90 = lengths.quantile(0.9)
        p90_word_count = int(round(p90))
        print(f"90th percentile word count: {p90_word_count}")

        sorted_lengths = lengths.sort_values()
        lower_mask = sorted_lengths <= p90
        upper_mask = sorted_lengths >= p90

        lower_index = lower_mask[lower_mask].index[-1] if lower_mask.any() else None
        upper_index = upper_mask[upper_mask].index[0] if upper_mask.any() else None

        if lower_index is not None:
            prompt = df.loc[lower_index, "prompt"]
            print("Prompt at or just below 90th percentile:")
            print(f"  length={lengths.loc[lower_index]} words")
            print(prompt)
        if upper_index is not None and upper_index != lower_index:
            prompt = df.loc[upper_index, "prompt"]
            print("Prompt at or just above 90th percentile:")
            print(f"  length={lengths.loc[upper_index]} words")
            print(prompt)

        if target_words is not None:
            diffs = (lengths - target_words).abs()
            matches = diffs[diffs <= tolerance]
            if matches.empty:
                print(
                    f"No prompts found within ±{tolerance} words of target ({target_words}) in {path.name}."
                )
            else:
                print(
                    f"Prompts within ±{tolerance} words of target ({target_words}):"
                )
                closest = matches.sort_values().head(max_prompts)
                for rank, idx in enumerate(closest.index, start=1):
                    prompt = df.loc[idx, "prompt"]
                    length = lengths.loc[idx]
                    print(f"  #{rank}: {length} words (diff {abs(length - target_words)})")
                    print(prompt)

        print()

        lengths_by_label.append((path.name, lengths))

    plot_histograms(lengths_by_label, bins=bins, output=output)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    csv_paths = [Path(p).expanduser().resolve() for p in args.files]
    missing: List[Path] = [p for p in csv_paths if not p.exists()]
    if missing:
        print("The following files do not exist:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        return 1

    analyze(
        csv_paths,
        bins=args.bins,
        output=args.output,
        target_words=args.target_words,
        tolerance=args.tolerance,
        max_prompts=args.max_prompts,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

