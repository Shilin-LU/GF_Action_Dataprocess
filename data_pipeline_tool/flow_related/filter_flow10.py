#!/usr/bin/env python3
"""
根据flow字段过滤JSON文件，只保留flow<=10的entry，
然后重新整理，每1000个entry放一个JSON文件。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List


DEFAULT_INPUT_DIRS = [
    Path("/share/project/denghaoge/shilinlu/dataset/GF-Minecraft/data_2003_tag"),
    Path("/share/project/denghaoge/shilinlu/dataset/GF-Minecraft/data_269_tag"),
]
DEFAULT_OUTPUT_DIR = Path("/share/project/denghaoge/shilinlu/dataset/GF-Minecraft/data_flow10_tag")
ENTRIES_PER_FILE = 1000
FLOW_THRESHOLD = 10.0


def discover_json_files(directories: List[Path]) -> Iterable[Path]:
    """发现所有JSON文件并按文件名排序。"""
    for directory in directories:
        if not directory.exists():
            print(f"[WARN] 目录不存在: {directory}")
            continue
        yield from sorted(directory.glob("*.json"))


def filter_entries_by_flow(data: List[dict], flow_threshold: float) -> List[dict]:
    """过滤出flow<=threshold的entry。"""
    filtered = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        
        flow = entry.get("flow")
        if flow is None:
            continue
        
        try:
            flow_value = float(flow)
            if flow_value <= flow_threshold:
                filtered.append(entry)
        except (ValueError, TypeError):
            continue
    
    return filtered


def save_entries_to_file(entries: List[dict], output_path: Path) -> None:
    """将entry列表保存到JSON文件。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False)


def main() -> None:
    """主函数。"""
    # 创建输出目录
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 收集所有符合条件的entry
    all_filtered_entries: List[dict] = []
    
    total_files = 0
    total_entries = 0
    total_filtered = 0
    
    print("开始扫描JSON文件...")
    for json_file in discover_json_files(DEFAULT_INPUT_DIRS):
        try:
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                print(f"[WARN] {json_file} 不是列表格式，跳过")
                continue
            
            total_files += 1
            total_entries += len(data)
            
            filtered = filter_entries_by_flow(data, FLOW_THRESHOLD)
            total_filtered += len(filtered)
            all_filtered_entries.extend(filtered)
            
            if total_files % 10 == 0:
                print(f"已处理 {total_files} 个文件, 收集 {total_filtered} 个符合条件的entry")
        
        except Exception as e:
            print(f"[ERROR] 处理文件 {json_file} 时出错: {e}")
            continue
    
    print(f"\n扫描完成:")
    print(f"  总文件数: {total_files}")
    print(f"  总entry数: {total_entries}")
    print(f"  符合条件的entry数: {total_filtered} (flow <= {FLOW_THRESHOLD})")
    
    # 将过滤后的entry分批保存
    print(f"\n开始保存到输出目录: {DEFAULT_OUTPUT_DIR}")
    
    file_index = 0
    for i in range(0, len(all_filtered_entries), ENTRIES_PER_FILE):
        batch = all_filtered_entries[i:i + ENTRIES_PER_FILE]
        output_filename = f"{file_index:06d}.json"
        output_path = DEFAULT_OUTPUT_DIR / output_filename
        
        save_entries_to_file(batch, output_path)
        file_index += 1
        
        if (i // ENTRIES_PER_FILE + 1) % 10 == 0:
            print(f"  已保存 {file_index} 个文件...")
    
    print(f"\n完成! 共生成 {file_index} 个JSON文件")
    print(f"输出目录: {DEFAULT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()

