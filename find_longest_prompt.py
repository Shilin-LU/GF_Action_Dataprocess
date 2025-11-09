#!/usr/bin/env python3
"""
Scan a CSV file containing prompts and report the row with the longest prompt.

Example:
    python find_longest_prompt.py --csv /share/project/denghaoge/shilinlu/dataset/GF-Minecraft/data_269/annotation_action.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable


@dataclass
class PromptStats:
    line_number: int
    prompt: str
    char_count: int
    token_count: int


def build_token_counter() -> Callable[[str], int]:
    """
    Try to construct a token counting function using tiktoken.
    Fallback to a whitespace-based tokenizer if tiktoken is unavailable.
    """
    try:
        import tiktoken

        @lru_cache(maxsize=1)
        def _encoding() -> "tiktoken.Encoding":
            try:
                return tiktoken.get_encoding("cl100k_base")
            except KeyError:
                return tiktoken.encoding_for_model("gpt-3.5-turbo")

        def count_tokens(text: str) -> int:
            return len(_encoding().encode(text))

        # Test the tokenizer once to surface any runtime errors early.
        _ = count_tokens("")
        return count_tokens

    except Exception:
        import re

        token_pattern = re.compile(r"\S+")

        def count_tokens(text: str) -> int:
            return len(token_pattern.findall(text))

        return count_tokens


def find_longest_prompt(csv_path: str, prompt_column: str = "prompt") -> PromptStats:
    token_counter = build_token_counter()
    longest: PromptStats | None = None

    try:
        with open(csv_path, newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            if prompt_column not in reader.fieldnames:
                raise ValueError(
                    f"Column '{prompt_column}' not found in CSV header: {reader.fieldnames}"
                )

            for idx, row in enumerate(reader, start=2):  # account for header line
                prompt = row.get(prompt_column, "") or ""
                char_count = len(prompt)
                token_count = token_counter(prompt)

                if (
                    longest is None
                    or char_count > longest.char_count
                    or (char_count == longest.char_count and token_count > longest.token_count)
                ):
                    longest = PromptStats(
                        line_number=idx,
                        prompt=prompt,
                        char_count=char_count,
                        token_count=token_count,
                    )
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"CSV file not found: {csv_path}") from exc

    if longest is None:
        raise ValueError("No prompt rows found in the provided CSV.")

    return longest


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Find the longest prompt in a CSV file and report its statistics."
    )
    parser.add_argument("--csv", required=True, help="Absolute path to the annotation_action.csv file.")
    parser.add_argument(
        "--prompt-column",
        default="prompt",
        help="Name of the column that contains the prompt text (default: 'prompt').",
    )
    args = parser.parse_args(argv)

    stats = find_longest_prompt(args.csv, args.prompt_column)

    print(f"Longest prompt is on line {stats.line_number}")
    print(f"Character count: {stats.char_count}")
    print(f"Token count: {stats.token_count}")
    print("Prompt content:")
    print(stats.prompt)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


