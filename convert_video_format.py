import argparse
import json
import subprocess
from pathlib import Path
from typing import Iterable, List, Sequence

DEFAULT_VIDEO_PATH = Path("/share/project/denghaoge/shilinlu/dataset/GF-Minecraft/data_2003/video/seed_1_part_1.mp4")
DEFAULT_JSON_PATH = Path("/share/project/denghaoge/shilinlu/dataset/GF-Minecraft/data_2003/json/000000.json")
DEFAULT_OUTPUT_ROOT = Path("/share/project/denghaoge/shilinlu/dataset/GF-Minecraft/frames")


def ensure_segments(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError("Unexpected JSON structure: expected list or dict.")


def filter_segments(segments: Sequence[dict], ids: Iterable[str] = None, indices: Iterable[int] = None) -> List[dict]:
    filtered = list(segments)
    if ids:
        id_set = set(ids)
        filtered = [s for s in filtered if s.get("id") in id_set]
    if indices:
        index_set = set(indices)
        filtered = [s for idx, s in enumerate(filtered) if idx in index_set]
    return filtered


def extract_segment(segment, output_root: Path, default_video: Path):
    start = int(segment["start"])
    end = int(segment["end"])
    if end < start:
        raise ValueError(f"Invalid segment with end < start: {segment}")

    video = Path(segment.get("video") or default_video)
    segment_id = segment.get("id") or f"{video.stem}_{start}_{end}"

    segment_dir = output_root / segment_id
    segment_dir.mkdir(parents=True, exist_ok=True)

    frame_pattern = segment_dir / "frame_%05d.png"
    frame_pattern_str = str(frame_pattern)

    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(video),
        "-vf", f"select='between(n,{start},{end})'",
        "-vsync", "0",
        "-start_number", str(start),
        frame_pattern_str
    ], check=True)

    output_video = segment_dir / f"{segment_id}.mp4"

    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", "16",
        "-start_number", str(start),
        "-i", frame_pattern_str,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-crf", "18",
        str(output_video)
    ], check=True)

    print(f"Segment {segment_id}: frames {start}-{end} saved to {segment_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract specified frame ranges from a video based on JSON metadata.")
    parser.add_argument("--json", default=str(DEFAULT_JSON_PATH), help="Path to the JSON file defining segments.")
    parser.add_argument("--video", default=str(DEFAULT_VIDEO_PATH), help="Default video path if a segment does not specify one.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_ROOT), help="Output directory for frames and rebuilt videos.")
    parser.add_argument("--ids", nargs="*", help="Segment IDs to extract (processes only these IDs).")
    parser.add_argument("--indexes", nargs="*", type=int, help="Segment indexes (0-based) to extract.")
    args = parser.parse_args()

    json_path = Path(args.json)
    default_video = Path(args.video)
    output_root = Path(args.output)

    output_root.mkdir(parents=True, exist_ok=True)

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    segments = ensure_segments(data)
    segments = filter_segments(segments, ids=args.ids, indices=args.indexes)

    if not segments:
        raise SystemExit("No segments matched the provided filters.")

    for segment in segments:
        extract_segment(segment, output_root=output_root, default_video=default_video)

    print("Done!")


if __name__ == "__main__":
    main()