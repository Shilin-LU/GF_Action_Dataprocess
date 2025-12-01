'''
python create_action_prompts.py --format combined --data-dir /share/project/denghaoge/shilinlu/dataset/GF-Minecraft/data_269
'''
import argparse
import csv
import json
import re
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List

DEFAULT_START_OFFSET = 15
DEFAULT_END_OFFSET = 17
DEFAULT_PITCH_THRESHOLD = 0.03
DEFAULT_YAW_THRESHOLD = 0.03

WS_LABELS = {0: "none", 1: "forward", 2: "backward"}
AD_LABELS = {0: "none", 1: "left", 2: "right"}
SCS_LABELS = {0: "none", 1: "jump", 2: "sneak", 3: "sprint"}

METADATA_NAME_PATTERN = re.compile(r"^(?P<prefix>seed_(?P<seed>\d+)_part)$")
PITCH_VALUE_KEY = "pitch_delta"
YAW_VALUE_KEY = "yaw_delta"


class OutputFormat(str, Enum):
    LABELS = "labels"
    VALUES = "values"
    COMBINED = "combined"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create annotation_action.csv by appending frame-wise action summaries "
            "to each caption in annotation.csv."
        )
    )
    default_data_dir = Path(__file__).resolve().parent / "data_269"
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir,
        help=f"Path to the data split directory (default: {default_data_dir})",
    )
    parser.add_argument(
        "--start-offset",
        type=int,
        default=DEFAULT_START_OFFSET,
        help="Number of frames to add to each start frame index before sampling actions.",
    )
    parser.add_argument(
        "--end-offset",
        type=int,
        default=DEFAULT_END_OFFSET,
        help="Number of frames to subtract from each end frame index before sampling actions.",
    )
    parser.add_argument(
        "--pitch-threshold",
        type=float,
        default=DEFAULT_PITCH_THRESHOLD,
        help="Threshold applied to pitch_delta to classify up/down/level.",
    )
    parser.add_argument(
        "--yaw-threshold",
        type=float,
        default=DEFAULT_YAW_THRESHOLD,
        help="Threshold applied to yaw_delta to classify left/right/steady.",
    )
    parser.add_argument(
        "--format",
        type=OutputFormat,
        choices=list(OutputFormat),
        default=OutputFormat.LABELS,
        help="Output format for appended actions (labels, values, combined).",
    )
    return parser.parse_args()


def ensure_terminal_period(text: str) -> str:
    stripped = text.rstrip()
    if not stripped.endswith("."):
        return stripped + "."
    return stripped


def classify_pitch(delta: float, threshold: float) -> str:
    if delta > threshold:
        return "up"
    if delta < -threshold:
        return "down"
    return "level"


def classify_yaw(delta: float, threshold: float) -> str:
    if delta > threshold:
        return "right"
    if delta < -threshold:
        return "left"
    return "steady"


def resolve_metadata_path(metadata_dir: Path, original_video: str) -> Path:
    stem = Path(original_video).stem  # e.g. seed_186_part
    match = METADATA_NAME_PATTERN.match(stem)
    if not match:
        raise ValueError(f"Unexpected video name format: {original_video}")
    seed_id = match.group("seed")
    metadata_name = f"{match.group('prefix')}_{seed_id}.json"
    metadata_path = metadata_dir / metadata_name
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file {metadata_path} not found for video {original_video}"
        )
    return metadata_path


def load_actions(metadata_path: Path) -> Dict[int, Dict]:
    with metadata_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    actions = data.get("actions", {})
    return {int(k): v for k, v in actions.items()}


def build_action_sequence(
    actions: Dict[int, Dict],
    frame_indices: Iterable[int],
    pitch_threshold: float,
    yaw_threshold: float,
    output_format: OutputFormat,
) -> List[str]:
    sequence = []
    frame_counter = 0
    for frame_index in frame_indices:
        action = actions.get(frame_index)
        if action is None:
            continue
        ws_raw = int(action.get("ws", 0))
        ad_raw = int(action.get("ad", 0))
        scs_raw = int(action.get("scs", 0))
        pitch_value = float(action.get(PITCH_VALUE_KEY, 0.0))
        yaw_value = float(action.get(YAW_VALUE_KEY, 0.0))
        ws_label = WS_LABELS.get(ws_raw, "none")
        ad_label = AD_LABELS.get(ad_raw, "none")
        scs_label = SCS_LABELS.get(scs_raw, "none")
        pitch_label = classify_pitch(pitch_value, pitch_threshold)
        yaw_label = classify_yaw(yaw_value, yaw_threshold)

        if output_format == OutputFormat.VALUES:
            entry = (
                f"F{frame_counter}[{ws_raw}; {ad_raw}; {scs_raw}; {pitch_value:.2f}; {yaw_value:.2f}]"
            )
        elif output_format == OutputFormat.COMBINED:
            entry = (
                f"F{frame_counter}[{ws_label}; {ad_label}; {scs_label}; {pitch_value:.2f}; {yaw_value:.2f}]"
            )
        else:
            entry = (
                f"F{frame_counter}[{ws_label}; {ad_label}; {scs_label}; {pitch_label}; {yaw_label}]"
            )
        sequence.append(entry)
        frame_counter += 1
    return sequence


def main() -> None:
    args = parse_args()
    data_dir: Path = args.data_dir.resolve()
    annotation_path = data_dir / "annotation.csv"
    output_path = data_dir / "annotation_action.csv"
    metadata_dir = data_dir / "metadata"

    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
    if not metadata_dir.is_dir():
        raise FileNotFoundError(f"Metadata directory not found: {metadata_dir}")

    metadata_cache: Dict[str, Dict[int, Dict]] = {}

    with annotation_path.open("r", encoding="utf-8") as annotation_file, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as output_file:
        reader = csv.DictReader(annotation_file)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {annotation_path}")

        writer = csv.DictWriter(output_file, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            original_video = row["original video name"]
            try:
                start_index = int(row["start frame index"])
                end_index = int(row["end frame index"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid frame indices in row: {row}"
                ) from exc

            adjusted_start = start_index + args.start_offset
            adjusted_end = end_index - args.end_offset

            frame_indices = range(adjusted_start, adjusted_end + 1)
            if adjusted_start > adjusted_end:
                frame_indices = []

            if original_video not in metadata_cache:
                metadata_path = resolve_metadata_path(metadata_dir, original_video)
                metadata_cache[original_video] = load_actions(metadata_path)

            actions_sequence = build_action_sequence(
                metadata_cache[original_video],
                frame_indices,
                args.pitch_threshold,
                args.yaw_threshold,
                args.format,
            )

            base_prompt = ensure_terminal_period(row["prompt"])
            if actions_sequence:
                action_text = "; ".join(actions_sequence)
            else:
                action_text = "No frame-aligned action data available."

            action_text = ensure_terminal_period(action_text)

            if args.format == OutputFormat.VALUES:
                header_text = "[ws;ad;scs;pitch;yaw]"
            else:
                header_text = "[forward; sideways; action; pitch; yaw]"

            action_summary = f"Actions per frame: {header_text}. {action_text}"
            row["prompt"] = f"{action_summary} {base_prompt}"
            writer.writerow(row)

    print(f"Wrote enriched annotations to {output_path}")


if __name__ == "__main__":
    main()
