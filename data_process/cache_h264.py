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
"""Cache H264 latents."""

import argparse
import json
import os
import shutil
import tempfile
import collections

import codewithgpu
from nvidia import dali
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
import torch

from diffnext.models.build import build_vae


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Build videos cache.")
    parser.add_argument("--jsons", type=str, required=True, help="Ground-truth json file.")
    parser.add_argument("--record", type=str, help="path to store record files")
    parser.add_argument("--vae", type=str, help="path to VAE model")
    parser.add_argument("--start", type=int, default=0, help="VAE inference batch size")
    parser.add_argument("--end", type=int, default=-1, help="VAE inference batch size")
    return parser.parse_args()


@dali.pipeline_def()
def dali_crop_pipeline(
    dataset,
    sequence_length=29,
    resize=256,
    crop_size=(256, 256),
    stride=6,
    image_type="RGB",
    dtype="float16",
    layout="CFHW",
    backend="gpu",
    mean=(127.5, 127.5, 127.5),
    std=(127.5, 127.5, 127.5),
):
    interp_type = dali.types.DALIInterpType.INTERP_TRIANGULAR
    image_type = getattr(dali.types.DALIImageType, image_type.upper())
    dali_dtype = getattr(dali.types.DALIDataType, dtype.upper(), None)
    video_args = {"file_list_include_preceding_frame": False, "file_list_frame_num": True}
    reader_args = {"device": backend, "image_type": image_type, "name": "reader"}
    reader_args = {"sequence_length": sequence_length, "stride": stride, **reader_args}
    reader_args = {"file_list": dataset, "skip_vfr_check": True, **video_args, **reader_args}
    reader_args = {"prefetch_queue_depth": 8, **reader_args}  # Faster?
    resize_args = {"resize_shorter": resize, "interp_type": interp_type}
    norm_args = {"mean": mean, "std": std, "dtype": dali_dtype}
    crop_norm_args = {**norm_args, **{"crop": crop_size, "output_layout": layout}}
    vid, idx = dali.fn.readers.video_resize(**resize_args, **reader_args)
    return dali.fn.crop_mirror_normalize(vid, **crop_norm_args), idx


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


def dump_videos(dataset, anns, writer, vae, stride=2, seqlen=24):
    # img_args, batch_size = {"resize": 480, "crop_size": (480, 768)}, 1
    # img_args, batch_size = {"resize": 256, "crop_size": (256, 384)}, 8
    img_args, batch_size = {"resize": 256, "crop_size": (256, 256)}, 8
    vid_args = {"stride": stride, "sequence_length": seqlen}
    pipe_args = {"batch_size": batch_size, "num_threads": 8, "device_id": vae.device.index}
    iterator_args = {"reader_name": "reader", "last_batch_policy": LastBatchPolicy.PARTIAL}
    pipe = dali_crop_pipeline(dataset, **img_args, **vid_args, **pipe_args)
    iterator = DALIGenericIterator(pipe, ["vid", "idx"], **iterator_args)
    while True:
        try:
            inputs = iterator.next()[0]
            batch_anns = [anns[i] for i in inputs["idx"].flatten().tolist()]
            batch_ids = [ann["id"] for ann in batch_anns]
            batch_y1 = [ann["text"] for ann in batch_anns]
            batch_y2 = [ann["caption"] for ann in batch_anns]
            batch_flow = [ann["flow"] for ann in batch_anns]
            batch_x = vae.encode(inputs["vid"]).latent_dist.parameters.cpu().numpy()
            for uid, x, y1, y2, flow in zip(batch_ids, batch_x, batch_y1, batch_y2, batch_flow):
                example = {"id": uid, "shape": x.shape, "text": y1, "caption": y2, "flow": flow}
                writer.write({"moments": x.tobytes(), **example})
        except StopIteration:
            break


if __name__ == "__main__":
    args = parse_args()

    rank, world_size = 0, 1
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()

    json_list = [args.jsons + "/" + x for x in os.listdir(args.jsons)]
    json_list.sort()
    error_set = None

    keep_json_list = []
    for json_file in json_list:
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
    vae = build_vae(args.vae, use_safetensors=False)
    vae = vae.to(device=device, dtype=dtype).eval()
    inference_mode = torch.inference_mode()
    inference_mode.__enter__()

    features = {"moments": "bytes", "caption": "string", "text": "string", "flow": "float64"}
    features = {"id": "string", "shape": ["int64"], **features}

    for json_file in rank_json_list:
        dataset, anns = load_json_dataset(json_file, error_set)
        record_path = os.path.join(args.record, json_file.split("/")[-1].split(".json")[0])
        if os.path.exists(record_path + "/METADATA"):
            continue
        shutil.rmtree(record_path) if os.path.exists(record_path) else None
        os.makedirs(record_path)
        writer = codewithgpu.RecordWriter(record_path, features, zfill_width=6)
        try:
            dump_videos(dataset.name, anns, writer, vae)
            print("Rank{}: Write {}.".format(rank, json_file))
            writer.close()
        except RuntimeError as e:
            print("Failed to dump ", json_file, e)
            writer.close()
            shutil.rmtree(record_path)
    print("Rank{}: Done.".format(rank))
