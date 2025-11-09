# Copyright (c) 2023-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
"""Merge record files."""

import os
import shutil
import json


def collect_recrods(record_path, features):
    all_paths = [os.path.join(record_path, x) for x in os.listdir(record_path)]
    _, all_data_files = all_paths.sort(), []
    for path in all_paths:
        if not os.path.isdir(path):
            continue
        data_files = [os.path.join(path, x) for x in os.listdir(path) if x.endswith(".data")]
        data_files.sort(), all_data_files.extend(data_files)
    num_entries, empty_dirs = 0, set()
    for src1 in all_data_files:
        name = src1.split("/")[-2]
        src2 = src1.replace(".data", ".index")
        num_entries += len(open(src2).readlines())
        shutil.move(src1, os.path.join(record_path, name + ".data"))
        shutil.move(src2, os.path.join(record_path, name + ".index"))
        empty_dirs.add(os.path.dirname(src1))
    [shutil.rmtree(path) for path in empty_dirs]
    with open(os.path.join(record_path, "METADATA"), "w") as f:
        json.dump({"entries": num_entries, "features": features}, f)


features = {
    "id": "STRING",
    "shape": ["INT64"],
    "codes": "BYTES",
    "caption": "STRING",
    "text": "STRING",
    "flow": "FLOAT32",
}
collect_recrods("/share/project/denghaoge/shilinlu/dataset/GF-Minecraft/GF_269_cache", features)

# features = {
#     "id": "STRING",
#     "text": {"data": "BYTES", "shape": ["INT64"]},
#     "caption": {"data": "BYTES", "shape": ["INT64"]},
# }
# collect_recrods("/share/project/panting/share_datasets/pt52_phi256_train", features)
