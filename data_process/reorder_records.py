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
"""Reorder record files."""

import os
import shutil
import json
import multiprocessing as mp

import codewithgpu


num = 0


def set_data_file(dataset, data_file):
    dataset._data_files, dataset._indices, dataset._size = [data_file], [], 0
    dataset._cursor, dataset._shard_id = 0, None
    if dataset._shard_loader is not None:
        dataset._shard_loader.close()
        dataset._shard_loader = None
    with open(data_file.replace(".data", ".index"), "r") as f:
        lines = f.readlines()
        dataset._size += len(lines)
        for line in lines:
            pos, size = line.split()
            dataset._indices.append((int(pos), int(size), 0))


def reorder(record_path):
    global num
    new_path = record_path.replace("_fix", "_fix2")
    if os.path.exists(new_path):
        return
    os.makedirs(new_path)
    dataset = codewithgpu.RecordDataset(record_path)
    num += len(dataset)
    writer = codewithgpu.RecordWriter(new_path, dataset._features, zfill_width=6)
    ids2example = dict((example["id"], example) for example in dataset)
    ids2example = sorted(ids2example.items())
    print(ids2example[0][0], ids2example[-1][0], record_path)
    for example_id, example in ids2example:
        writer.write(example)
    writer.close()


def reorder_v2(data_file, dataset):
    global num
    new_path = data_file.replace("_fix", "_fix2").replace(".data", "")
    if os.path.exists(new_path):
        return
    os.makedirs(new_path)
    set_data_file(dataset, data_file)
    num += len(dataset)
    writer = codewithgpu.RecordWriter(new_path, dataset._features, zfill_width=6)
    ids2example = dict((example["id"], example) for example in dataset)
    ids2example = sorted(ids2example.items())
    print(ids2example[0][0], ids2example[-1][0], data_file)
    for example_id, example in ids2example:
        writer.write(example)
    writer.close()


def actor_fn(q):
    dataset = codewithgpu.RecordDataset(path)
    while True:
        data_file = q.get()
        if data_file is None:
            break
        new_path = data_file.replace("_fix", "_fix2").replace(".data", "")
        # if os.path.exists(new_path):
        #     return
        os.makedirs(new_path)
        set_data_file(dataset, data_file)
        writer = codewithgpu.RecordWriter(new_path, dataset._features, zfill_width=6)
        ids2example = dict((example["id"], example) for example in dataset)
        ids2example = sorted(ids2example.items())
        print(ids2example[0][0], ids2example[-1][0], data_file)
        for example_id, example in ids2example:
            writer.write(example)
        writer.close()


path = "/share/project/panting/share_datasets/emu3_cog480/stock_flow30_fix/"
records = [path + x for x in os.listdir(path) if x.endswith(".data")]
records.sort()

# dataset = codewithgpu.RecordDataset(path)
# set_data_file(dataset, "/share/project/panting/share_datasets/emu3_cog480/stock_1072580_fix/000002.data")
# ids2example = dict((example["id"], example) for example in dataset)
# print(len(ids2example))
# # for example in dataset:
# #     print(example["id"])


# records = []
# root = "/share/project/panting/share_datasets/emu3_cog480/stock_1072580_fix2/"
# for i in range(1073):
#     d = root + str(i).zfill(6) + ".data"
#     if not os.path.exists(d):
#       records.append(path + str(i).zfill(6) + ".data")
#       print(records[-1])

num_actors = 16
q = mp.Queue()
actors = [mp.Process(target=actor_fn, args=(q,)) for _ in range(num_actors)]
[actor.start() for actor in actors]
for data_file in records:
    q.put(data_file)
[q.put(None) for actor in actors], [actor.join() for actor in actors]

# dataset = codewithgpu.RecordDataset(path)
# records.sort(), [reorder_v2(x, dataset) for x in records]
# print(num)
