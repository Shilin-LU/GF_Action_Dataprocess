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
"""Cache panda video latents."""

import argparse
import json
import numpy as np
import multiprocessing as mp
import os
import tempfile
import time
import cv2

import torch
from ram.models import ram_plus


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Build videos cache.")
    parser.add_argument("--jsons", type=str, help="Ground-truth json file.")
    parser.add_argument("--record", type=str, help="path to store record files")
    parser.add_argument("--model", type=str, help="path to TAG model")
    parser.add_argument("--start", type=int, default=0, help="VAE inference batch size")
    parser.add_argument("--end", type=int, default=-1, help="VAE inference batch size")
    return parser.parse_args()


def load_json_dataset(json_file, error_set=None):
    """Load video json annotations."""
    anns = json.load(open(json_file))
    dataset = tempfile.NamedTemporaryFile(mode="w", suffix=".txt")
    for ann_index, ann in enumerate(anns):
        if error_set and ann["video"] in error_set:
            continue
        line = "{} {}".format(ann["video"], ann_index)
        if ann["start"] is not None:
            line += " {} {}".format(int(ann["start"]), int(ann["end"]))
        dataset.write(line + "\n")
    dataset.flush()
    return dataset, anns


def actor_fn(input_queue, output_queue, num_frames=9, resize=384):
    import decord  # safe import.

    while True:
        index, line = input_queue.get()
        if index is None:
            break
        parts = line.strip().split()
        video_path = line.strip()[: -len(" " + " ".join(parts[-3:]))]
        lbl, st, ed = int(parts[-3]), int(parts[-2]), int(parts[-1])
        # frame_ids = np.linspace(st, ed - 1, num=num_frames + 2)[1:-1].tolist()
        frame_ids = np.linspace(st, ed - 1, num=num_frames).tolist()
        try:
            reader = decord.VideoReader(video_path, height=resize, width=resize)
            frames = reader.get_batch(frame_ids).asnumpy()
            output_queue.put((index, frames, lbl))
        except Exception as e:
            print(e)
            output_queue.put((index, None, lbl))


def calculate_flow(img_pre, img_next):
    # Convert images to grayscale
    gray_pre = cv2.cvtColor(img_pre, cv2.COLOR_RGB2GRAY)
    gray_next = cv2.cvtColor(img_next, cv2.COLOR_RGB2GRAY)
    # Detect feature points in the first frame
    feature_params = dict(maxCorners=100, qualityLevel=0.1, blockSize=7, minDistance=7)
    p0 = cv2.goodFeaturesToTrack(gray_pre, mask=None, **feature_params)
    if p0 is None or len(p0) == 0:
        return 0.0
    # Calculate optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2)
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray_pre, gray_next, p0, None, **lk_params)
    if st is None or len(st) == 0:
        return 0.0
    good_new = p1[st.ravel() == 1]
    good_old = p0[st.ravel() == 1]
    if len(good_new) == 0:
        return 0.0
    # 使用欧氏距离
    displacements = np.linalg.norm(good_new - good_old, axis=1)
    valid_dists = displacements[displacements > 2.0]  # 只保留大于2像素的运动
    return float(np.mean(valid_dists)) if len(valid_dists) > 0 else 0.0


def dump_batch(batch_ids, batch_vid, anns, model):
    batch_flows = []
    for vid in batch_vid:
        batch_flows.append([calculate_flow(vid[i], vid[i + 1]) for i in range(len(vid) - 1)])
    x = torch.as_tensor(np.stack(batch_vid)[:, 1::2], device=model.device).to(model.dtype)
    assert x.shape[1] == 4, x.shape
    stdv = torch.as_tensor([58.395, 57.12, 57.375]).to(x)
    mean = torch.as_tensor([123.675, 116.28, 103.53]).to(x)
    x, tag_shape = x.sub_(mean).div_(stdv).flatten(0, 1).permute(0, 3, 1, 2), x.shape[:2]
    batch_tags = np.array(model.generate_tag(x)[0]).reshape(tag_shape)
    for line_id, tags, flows in zip(batch_ids, batch_tags, batch_flows):
        anns[line_id]["tags"] = tags.tolist()
        anns[line_id]["flows"] = flows
    return [], []


def dump_videos(dataset, anns, model, out_file):
    num_readers, lines = 16, open(dataset).readlines()
    batch_size = 32
    q1, q2 = mp.Queue(), mp.Queue(128)
    actors = [mp.Process(target=actor_fn, args=(q1, q2)) for _ in range(num_readers)]
    [actor.start() for actor in actors]
    [q1.put((i, line)) for i, line in enumerate(lines)]
    batch_ids, batch_vid = [], []
    for i in range(len(lines)):
        index, frames, line_id = q2.get()
        if frames is None:
            print("Failed to decode", lines[index].strip().split()[0])
            continue
        batch_ids.append(line_id), batch_vid.append(frames)
        if len(batch_ids) != batch_size:
            continue
        batch_ids, batch_vid = dump_batch(batch_ids, batch_vid, anns, model)
    dump_batch(batch_ids, batch_vid, anns, model) if len(batch_ids) > 0 else None
    json.dump(anns, open(out_file, "w"))
    [q1.put((None, None)) for _ in range(num_readers)]
    [actor.join() for actor in actors]


if __name__ == "__main__":
    args = parse_args()

    rank, world_size = 0, 1
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()

    json_list = [args.jsons + "/" + x for x in os.listdir(args.jsons)]
    json_list.sort()

    keep_json_list = []
    for json_file in json_list:
        record_path = os.path.join(args.record, json_file.split("/")[-1].split(".json")[0])
        if os.path.exists(os.path.join(args.record, os.path.split(json_file)[-1])):
            continue
        keep_json_list.append(json_file)
    json_list = keep_json_list

    json_list = json_list[args.start : args.end] if args.end > 0 else json_list
    print("Load %d json files." % (len(json_list))) if rank == 0 else None
    rank_json_list = json_list[slice(rank, None, world_size)]
    print("Rank{}: {} json files for caching.".format(rank, len(rank_json_list)))

    device, dtype = torch.device("cuda", rank % 8), torch.float16
    model = ram_plus(pretrained=args.model, image_size=384, vit="swin_l")
    model = model.to(device).to(dtype).eval()
    model.device, model.dtype = device, dtype
    inference_mode = torch.inference_mode()
    inference_mode.__enter__()

    os.makedirs(args.record, exist_ok=True)
    for json_file in rank_json_list:
        dataset, anns = load_json_dataset(json_file)
        out_file = os.path.join(args.record, os.path.split(json_file)[-1])
        tic = time.time()
        dump_videos(dataset.name, anns, model, out_file)
        print("Rank{}: {}s Write {}.".format(rank, int(time.time() - tic), json_file))
