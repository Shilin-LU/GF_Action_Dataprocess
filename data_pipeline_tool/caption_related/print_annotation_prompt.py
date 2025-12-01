'''
python print_annotation_prompt.py 12 --csv /path/to/other.csv
'''
import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    default_csv = (
        Path(__file__).resolve().parent / "data_269" / "annotation_action.csv"
    )
    parser = argparse.ArgumentParser(
        description="Print the prompt of the nth data row in annotation_action.csv."
    )
    parser.add_argument(
        "row",
        type=int,
        help="Row number to display (1-based, excluding header).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=default_csv,
        help=f"Path to annotation_action.csv (default: {default_csv})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path: Path = args.csv.resolve()

    if args.row <= 0:
        raise ValueError("Row number must be a positive integer (1-based).")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    target_index = args.row

    with csv_path.open("r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for idx, row in enumerate(reader, start=1):
            if idx == target_index:
                prompt = row.get("prompt")
                if prompt is None:
                    raise KeyError(
                        f"'prompt' column not found in CSV header: {reader.fieldnames}"
                    )
                print(prompt)
                return

    raise IndexError(
        f"Row {target_index} out of range. CSV contains {idx if 'idx' in locals() else 0} data rows."
    )


if __name__ == "__main__":
    main()

