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
"""Cache video latents."""

import argparse
import json
import os
import shutil
import tempfile

import numpy as np
import codewithgpu
from nvidia import dali
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
import torch


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Build videos cache.")
    parser.add_argument("--jsons", type=str, required=True, help="Ground-truth json file.")
    parser.add_argument("--record", type=str, help="path to store record files")
    parser.add_argument("--vae", type=str, help="path to VAE model")
    parser.add_argument("--vae-batch-size", type=int, default=1, help="VAE inference batch size")
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
    reader_args = {"file_list": dataset, **video_args, **reader_args}
    resize_args = {"resize_shorter": resize, "interp_type": interp_type}
    norm_args = {"mean": mean, "std": std, "dtype": dali_dtype}
    crop_norm_args = {**norm_args, **{"crop": crop_size, "output_layout": layout}}
    vid, lbl = dali.fn.readers.video_resize(**resize_args, **reader_args)
    return dali.fn.crop_mirror_normalize(vid, **crop_norm_args), lbl


def load_json_dataset(json_file):
    """Load video json annotations."""
    cap_data_dict = json.load(open(json_file))
    dataset = tempfile.NamedTemporaryFile(mode="w", suffix=".txt")
    for index, data_dict in enumerate(cap_data_dict):
        line = "{} {}".format(data_dict["video"], index)
        if data_dict["start"] is not None:
            line += " {} {}".format(data_dict["start"], data_dict["end"])
        dataset.write(line + "\n")
    dataset.flush()
    return dataset, cap_data_dict


def dump_videos(dataset_file, captions):
    img_args = {"resize": 480, "crop_size": (480, 768), "dtype": "float16"}
    vid_args, batch_size = {"stride": 6, "sequence_length": 29}, 1
    pipe_args = {"batch_size": 1, "num_threads": 8, "device_id": 0}
    iterator_args = {"reader_name": "reader", "last_batch_policy": LastBatchPolicy.PARTIAL}
    pipe = dali_crop_pipeline(dataset_file, **img_args, **vid_args, **pipe_args)
    iterator = DALIGenericIterator(pipe, ["vid", "lbl"], **iterator_args)
    uid = set()
    for i in range(10):
        inputs = iterator.next()[0]
        if captions[int(inputs["lbl"])]["id"] in uid:
            continue
        uid.add(captions[int(inputs["lbl"])]["id"])
        data = {"video": inputs["vid"], "caption": captions[int(inputs["lbl"])]["text"]}
        torch.save(data, "video_data_%d.pth" % i)


if __name__ == "__main__":
    args = parse_args()

    json_list = [args.jsons + "/" + x for x in os.listdir(args.jsons)]
    json_list.sort()

    device, dtype = torch.device("cuda", 0), torch.float16
    inference_mode = torch.inference_mode()
    inference_mode.__enter__()

    features = {"moments": "bytes", "caption": "string", "text": "string"}
    features = {"id": "string", "shape": ["int64"], **features}

    for json_file in json_list:
        dataset, captions = load_json_dataset(json_file)
        dump_videos(dataset.name, captions)
        break
