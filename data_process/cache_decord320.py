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
import shutil
import tempfile

import codewithgpu
import torch


def build_vae(pretrained_path, **kwargs):
    """Create an VAE instance."""
    if not isinstance(pretrained_path, str):
        return pretrained_path

    from diffnext.models.autoencoders.autoencoder_kl import AutoencoderKL
    from diffnext.models.autoencoders.autoencoder_kl_cogvideox import AutoencoderKLCogVideoX
    from diffnext.models.autoencoders.autoencoder_kl_ltx import AutoencoderKLLTXVideo
    from diffnext.models.autoencoders.autoencoder_vq_cosmos3d import AutoencoderVQCosmos3D

    config_file = os.path.join(pretrained_path, "config.json")
    class_type = locals()[json.load(open(config_file))["_class_name"]]
    return class_type.from_pretrained(pretrained_path, **kwargs)


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Build videos cache.")
    parser.add_argument("--jsons", type=str, help="Ground-truth json file.")
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
        start_frame = ann.get("start")
        end_frame = ann.get("end")
        if start_frame is not None and end_frame is not None:
            frame_count = int(end_frame) - int(start_frame) + 1
            if frame_count <= 0:
                continue
            line += " {} {}".format(int(start_frame), frame_count)
        elif start_frame is not None:
            line += f" {int(start_frame)}"
        dataset.write(line + "\n")
    dataset.flush()
    return dataset, anns


def actor_fn(input_queue, output_queue, seqlen=49, resize=320, crop_size=(320, 512)):
    import decord  # safe import.

    while True:
        index, line = input_queue.get()
        if index is None:
            break
        parts = line.strip().split()
        video_path = line.strip()[: -len(" " + " ".join(parts[-3:]))]
        lbl = int(parts[-3])
        st = int(parts[-2])
        frame_count = int(parts[-1])
        if frame_count <= 0:
            output_queue.put((index, None, lbl))
            continue
        end = st + frame_count
        if seqlen > 1:
            stride = max((frame_count - 1) // (seqlen - 1), 1)
        else:
            stride = 1
        frame_ids = list(range(st, end, stride))
        frame_ids = frame_ids[:seqlen]
        if frame_ids and frame_ids[-1] != end - 1 and len(frame_ids) < seqlen:
            frame_ids.append(end - 1)
        frame_ids = frame_ids[:seqlen]
        try:
            reader = decord.VideoReader(video_path)
            h, w = reader[0].shape[:2]
            scale = float(resize) / float(min(h, w))
            size = int(h * scale + 0.5), int(w * scale + 0.5)
            y, x = (size[0] - crop_size[0]) // 2, (size[1] - crop_size[1]) // 2
            reader = decord.VideoReader(video_path, height=size[0], width=size[1])
            frames = reader.get_batch(frame_ids).asnumpy()
            frames = frames[:, y : y + crop_size[0], x : x + crop_size[1]]
            output_queue.put((index, frames, lbl))
        except Exception as e:
            print(e)
            output_queue.put((index, None, lbl))


def dump_batch(batch_ids, batch_vid, anns, writer, vae):
    x = torch.as_tensor(np.stack(batch_vid), device=vae.device).to(dtype=vae.dtype)
    x = x.permute(0, 4, 1, 2, 3).sub_(127.5).div_(127.5)
    batch_moments = vae.encode(x).latent_dist.parameters.cpu().numpy()
    for line_id, vid in zip(batch_ids, batch_moments):
        ann = anns[line_id]
        uid, flow, y1, y2 = ann["id"], ann["flow"], ann["text"], ann["caption"]
        if "flows" in ann:
            flow = np.mean(ann["flows"])
        else:
            print("Missing flows", ann["id"])
        example = {"id": uid, "shape": vid.shape, "text": y1, "caption": y2, "flow": flow}
        x_key = "codes" if vid.dtype == "int32" else "moments"
        writer.write({x_key: vid.tobytes(), **example})
    return [], []


def dump_videos(dataset, anns, writer, vae, seqlen=49):
    num_readers, lines = 16, open(dataset).readlines()
    img_args, batch_size = {"resize": 320, "crop_size": (320, 512)}, 4
    q1, q2 = mp.Queue(), mp.Queue(128)
    # fmt: off
    actors = [mp.Process(target=actor_fn, args=(
        q1, q2, seqlen,
        img_args["resize"], img_args["crop_size"])) for _ in range(num_readers)]
    # fmt: on
    [actor.start() for actor in actors]
    [q1.put((i, line)) for i, line in enumerate(lines)]
    batch_ids, batch_vid = [], []
    for i in range(len(lines)):
        index, frames, line_id = q2.get()
        if frames is None:
            print("Failed to decode", lines[index].strip().split()[0])
            continue
        if frames.shape != (49, 320, 512, 3):
            print("Bad frames", frames.shape, lines[index].strip().split()[0])
            continue
        batch_ids.append(line_id), batch_vid.append(frames)
        if len(batch_ids) != batch_size:
            continue
        batch_ids, batch_vid = dump_batch(batch_ids, batch_vid, anns, writer, vae)
    dump_batch(batch_ids, batch_vid, anns, writer, vae) if len(batch_ids) > 0 else None
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
    for i, json_file in enumerate(json_list):
        record_path = os.path.join(args.record, json_file.split("/")[-1].split(".json")[0])
        if os.path.exists(record_path + "/METADATA"):
            continue
        keep_json_list.append(json_file)
    json_list = keep_json_list

    json_list = json_list[args.start : args.end] if args.end > 0 else json_list
    print("Load %d json files." % (len(json_list))) if rank == 0 else None
    rank_json_list = json_list[slice(rank, None, world_size)]
    print("Rank{}: {} json files for caching.".format(rank, len(rank_json_list)))

    device, dtype = torch.device("cuda", rank % 8), torch.float16
    vae = build_vae(args.vae)
    vae = vae.to(device=device, dtype=dtype).eval()
    inference_mode = torch.inference_mode()
    inference_mode.__enter__()

    features = {"moments": "bytes", "caption": "string", "text": "string", "flow": "float64"}
    features = {"id": "string", "shape": ["int64"], **features}
    features.setdefault("codes", features.pop("moments")) if hasattr(vae, "quantizer") else None

    for json_file in rank_json_list:
        dataset, anns = load_json_dataset(json_file)
        record_path = os.path.join(args.record, json_file.split("/")[-1].split(".json")[0])
        if os.path.exists(record_path + "/METADATA"):
            continue
        shutil.rmtree(record_path) if os.path.exists(record_path) else None
        os.makedirs(record_path)
        writer = codewithgpu.RecordWriter(record_path, features, zfill_width=6)
        dump_videos(dataset.name, anns, writer, vae)
        print("Rank{}: Write {}.".format(rank, json_file))
        writer.close()
