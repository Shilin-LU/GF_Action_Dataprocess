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
import multiprocessing as mp
import os
import shutil
import tempfile

import codewithgpu
import torch

from diffnext.models.build import build_vae


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Build videos cache.")
    parser.add_argument("--jsons", type=str, required=True, help="Ground-truth json file.")
    parser.add_argument("--jsons2", type=str, required=True, help="Ground-truth json file.")
    parser.add_argument("--record", type=str, help="path to store record files")
    parser.add_argument("--vae", type=str, help="path to VAE model")
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


def actor_fn(input_queue, output_queue, stride=2, seqlen=29, resize=480, crop_size=(480, 640)):
    import decord  # safe import.

    while True:
        index, line = input_queue.get()
        if index is None:
            break
        parts = line.strip().split()
        frame_ids = list(range(int(parts[2]), int(parts[3]), stride))
        assert len(frame_ids) == seqlen, (len(frame_ids), seqlen)
        try:
            reader = decord.VideoReader(parts[0])
            h, w = reader[0].shape[:2]
            scale = float(resize) / float(min(h, w))
            size = int(h * scale + 0.5), int(w * scale + 0.5)
            y, x = (size[0] - crop_size[0]) // 2, (size[1] - crop_size[1]) // 2
            reader = decord.VideoReader(parts[0], height=size[0], width=size[1])
            frames = reader.get_batch(frame_ids).asnumpy()
            frames = frames[:, y : y + crop_size[0], x : x + crop_size[1]]
            output_queue.put((index, frames, int(parts[1])))
        except Exception as e:
            print(e)
            output_queue.put((index, None, int(parts[1])))


def dump_videos(dataset, anns, writer, vae, stride=2, seqlen=29):
    num_readers, lines = 12, open(dataset).readlines()
    img_args = {"resize": 480, "crop_size": (480, 640)}
    input_queues = [mp.Queue() for _ in range(num_readers)]
    output_queues = [mp.Queue(100) for _ in range(num_readers)]
    # fmt: off
    actors = [mp.Process(target=actor_fn, args=(
        input_queues[i], output_queues[i], stride, seqlen,
        img_args["resize"], img_args["crop_size"])) for i in range(num_readers)]
    # fmt: on
    [actor.start() for actor in actors]
    for i, line in enumerate(lines):
        input_queues[i % num_readers].put((i, line))
    for i in range(num_readers):
        input_queues[i].put((None, None))
    for i in range(len(lines)):
        index, frames, line_id = output_queues[i % num_readers].get()
        if frames is None:
            print("Failed to decode", lines[index].strip().split()[0])
            continue
        ann = anns[line_id]
        uid, flow, y1, y2 = ann["id"], ann["flow"], ann["text"], ann["caption"]
        x = torch.as_tensor(frames, device=vae.device).to(dtype=vae.dtype)
        x = x.permute(3, 0, 1, 2).unsqueeze_(0).sub_(127.5).div_(127.5)
        x = vae.encode(x).latent_dist.parameters.squeeze_(0).cpu().numpy()
        example = {"id": uid, "shape": x.shape, "text": y1, "caption": y2, "flow": flow}
        writer.write({"moments": x.tobytes(), **example})
    [actor.join() for actor in actors]


if __name__ == "__main__":
    args = parse_args()

    rank, world_size = 0, 1
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()

    json_list = [args.jsons + "/" + x for x in os.listdir(args.jsons)]
    json_list += [args.jsons2 + "/" + x for x in os.listdir(args.jsons2)]
    json_list.sort()
    json_list = json_list[args.start : args.end]
    print(f"strat : {args.start} -> end : {args.end}")

    print("Load %d json files." % (len(json_list))) if rank == 0 else None
    rank_json_list = json_list[slice(rank, None, world_size)]
    print("Rank{}: {} json files for caching.".format(rank, len(rank_json_list)))

    device, dtype = torch.device("cuda", rank % 8), torch.float16
    vae = AutoencoderKL.from_pretrained(args.vae, use_safetensors=False)
    vae = vae.to(device=device, dtype=dtype).eval()
    inference_mode = torch.inference_mode()
    inference_mode.__enter__()

    features = {"moments": "bytes", "caption": "string", "text": "string", "flow": "float64"}
    features = {"id": "string", "shape": ["int64"], **features}

    for json_file in rank_json_list:
        dataset, anns = load_json_dataset(json_file)
        record_path = os.path.join(args.record, json_file.split("/")[-1].split(".json")[0])
        shutil.rmtree(record_path) if os.path.exists(record_path) else None
        os.makedirs(record_path)
        writer = codewithgpu.RecordWriter(record_path, features, zfill_width=6)
        dump_videos(dataset.name, anns, writer, vae)
        print("Rank{}: Write {}.".format(rank, json_file))
        writer.close()
