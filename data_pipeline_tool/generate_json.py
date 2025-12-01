'''
python generate_json.py --data-root /share/project/denghaoge/shilinlu/dataset/GF-Minecraft/data_269
'''
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Set


def discover_metadata(metadata_dir: Path) -> Set[str]:
    metadata_basenames: Set[str] = set()
    for metadata_path in metadata_dir.glob("*.json"):
        stem = metadata_path.stem
        basename = re.sub(r"_\d+$", "", stem)
        metadata_basenames.add(basename)
    return metadata_basenames


def normalize_video_name(name: str) -> str:
    return re.sub(r"_[0-9]+(?=\.[^.]+$)", "", name)


def discover_video_paths(video_dirs: List[Path]) -> Dict[str, Path]:
    video_paths: Dict[str, Path] = {}
    for video_dir in video_dirs:
        if not video_dir.exists():
            continue
        for file_path in video_dir.rglob("*"):
            if file_path.is_file():
                filename = file_path.name
                video_paths.setdefault(filename, file_path)
                normalized = normalize_video_name(filename)
                video_paths.setdefault(normalized, file_path)
    return video_paths


def load_annotations(
    annotation_path: Path,
    metadata_basenames: Set[str],
    video_paths: Dict[str, Path],
) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    missing_metadata: Set[str] = set()
    missing_video: Set[str] = set()

    with annotation_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            original_name = row.get("original video name", "").strip()
            if not original_name:
                continue

            video_name = original_name.rsplit(".", 1)[0]
            if metadata_basenames and video_name not in metadata_basenames:
                missing_metadata.add(video_name)
                continue

            normalized_original = normalize_video_name(original_name)
            video_path = video_paths.get(original_name) or video_paths.get(
                normalized_original
            )
            if not video_path:
                missing_video.add(original_name)
                continue

            try:
                start = int(row.get("start frame index", "0"))
                end = int(row.get("end frame index", "0"))
            except ValueError:
                continue

            if end < start:
                start, end = end, start

            adjusted_start = start + 15
            adjusted_end = end - 17

            if adjusted_end < adjusted_start:
                continue

            caption = row.get("prompt", "").strip()
            entry_id = generate_entry_id(video_name, adjusted_start, adjusted_end)

            entries.append(
                {
                    "start": adjusted_start,
                    "end": adjusted_end,
                    "video": str(video_path),
                    "text": "",
                    "caption": caption,
                    "id": entry_id,
                }
            )

    if missing_metadata:
        print(
            f"跳过 {len(missing_metadata)} 个在 metadata 中未匹配到的原始视频: "
            + ", ".join(sorted(missing_metadata))
        )

    if missing_video:
        print(
            f"跳过 {len(missing_video)} 个在 video 目录中未找到文件的原始视频: "
            + ", ".join(sorted(missing_video))
        )

    return entries


def generate_entry_id(video_name: str, start: int, end: int) -> str:
    return f"{video_name}_{start}_{end}"


def write_json_chunks(entries: List[Dict[str, object]], output_dir: Path, chunk_size: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(0, len(entries), chunk_size):
        chunk = entries[idx : idx + chunk_size]
        chunk_index = idx // chunk_size
        output_path = output_dir / f"{chunk_index:06d}.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(chunk, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="根据 data_269 的 metadata 与 annotation.csv 生成分片 JSON"
    )
    default_data_root = Path(__file__).resolve().parent / "data_269"
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root,
        help="包含 metadata 目录与 annotation.csv 的数据根目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出 JSON 文件的目录，默认为 data-root 下的 json",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="单个输出 JSON 文件包含的最大条目数",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root: Path = args.data_root
    metadata_dir = data_root / "metadata"
    annotation_path = data_root / "annotation_action.csv"

    if not metadata_dir.exists():
        raise FileNotFoundError(f"未找到 metadata 目录: {metadata_dir}")
    if not annotation_path.exists():
        raise FileNotFoundError(f"未找到 annotation.csv: {annotation_path}")

    output_dir = args.output_dir or data_root / "json"

    metadata_basenames = discover_metadata(metadata_dir)
    if not metadata_basenames:
        raise RuntimeError("metadata 目录为空，无法生成 JSON")

    candidate_roots = {data_root}
    for dirname in ("data_269", "data_2003"):
        candidate = data_root.parent / dirname
        candidate_roots.add(candidate)

    video_dirs = []
    for root in candidate_roots:
        video_dir = root / "video"
        if video_dir.exists():
            video_dirs.append(video_dir)

    if not video_dirs:
        raise RuntimeError("未找到可用的 video 目录，无法匹配原始视频文件")

    video_paths = discover_video_paths(video_dirs)

    entries = load_annotations(annotation_path, metadata_basenames, video_paths)

    if not entries:
        print("没有可写入的条目")
        return

    write_json_chunks(entries, output_dir, args.chunk_size)
    print(
        f"生成完成：共 {len(entries)} 条记录，写入目录 {output_dir}，单文件最多 {args.chunk_size} 条"
    )


if __name__ == "__main__":
    main()
