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

import os
import numpy as np
import json
import tqdm
import collections


def load_and_split(json_file, seqlen=64, fps_range=(23.5, 30.5), stride=2, shape=(256, 384)):
    """Load video json annotations."""
    min_frame_count, base_size = (seqlen - 1) * stride + 1, shape[0]
    merge_data_dict, split_data_dict = json.load(open(json_file)), []
    for data_dict in tqdm.tqdm(merge_data_dict):
        info = data_dict["video_info"]
        if info["frame_count"] < min_frame_count:
            continue
        # if info["fps"] < fps_range[0] or info["fps"] > fps_range[1]:
        #     continue
        # if info["flow"] < 1.5:
        #     continue
        # if info["aes_score"][1] < 4.75:
        #     continue
        video_shape = info["shape"]
        try:
            scale = base_size / min(video_shape)
        except TypeError as e:
            continue
        if any(int(x * scale + 0.5) < y for (x, y) in zip(video_shape, shape)):
            continue
        for cap_dict in data_dict["captions"]:
            if cap_dict["start"] is None and cap_dict["end"] is None:
                cap_dict["start"], cap_dict["end"] = 0, info["frame_count"]
                assert info["frame_count"]
            st, ed = cap_dict["start"], cap_dict["end"]
            center_st = max((ed - st - min_frame_count), 0) // 2 + st
            center_ed = min(center_st + min_frame_count, ed)
            if center_ed - center_st < min_frame_count:
                continue
            assert center_ed - center_st == min_frame_count, (center_st, center_ed)
            cap_data_dict = {"start": center_st, "end": center_ed, "video": data_dict["path"]}
            cap_data_dict["count"] = ed
            # cap_data_dict["video"] = cap_data_dict["video"].replace(
            #     "/share/project/lzx/video_data", "/share/project/datasets"
            # )
            # cap_data_dict["video"] = cap_data_dict["video"].replace(
            #     "/share/project/lzx/video_data/gaopin600p", "/share/project/datasets/gaopin_600p"
            # )
            # cap_data_dict["video"] = cap_data_dict["video"].replace(
            #     "/share/project/lzx/video_data/gaopin_1080p", "/share/project/datasets/gaopin_1080p"
            # )
            cap_data_dict.update({"text": data_dict["description"], "caption": cap_dict["caption"]})
            split_data_dict += [{"flow": info.get("flow", 5), **cap_data_dict}]
    return split_data_dict


if __name__ == "__main__":
    np.random.seed(1337)
    json_files = []

    # fmt: off
    vid_args = {"seqlen": 49, "shape": (240, 384)}

    json_path = "/share/project/panting/share_datasets/raw_video_data/49x384x240/pexels/"
    json_files = ["/share/project/denghaoge/Autoregressive/diffnext_dhg_video/dataset/pexels/filtered_videos_pexels.json"]

    json_path = "/share/project/panting/share_datasets/raw_video_data/49x384x240/artgrid/"
    json_files = ["/share/project/denghaoge/Autoregressive/diffnext_dhg_video/dataset/artgrid/filtered_videos_artgrid.json"]

    # json_files = ["/share/project/denghaoge/nova/project/preprocess_data/json/motionarray/2_process_data_motionarray_flow_2_50_num_215883.json"]
    # json_files = ["/share/project/denghaoge/nova/project/preprocess_data/json/filmsupply/2_process_data_filmsupply_flow_2_50_num_446247.json"]

    # json_path = "/share/project/panting/share_datasets/raw_video_data/49x384x240/videvo_low/"
    # json_files = ["/share/project/denghaoge/nova/project/preprocess_data/json/videvo_low/2_process_data_videvo_low_flow_2_50_num_465325.json"]

    # json_path = "/share/project/panting/share_datasets/raw_video_data/45x384x240/videvo_high/"
    # json_files = ["/share/project/denghaoge/nova/project/preprocess_data/json/videvo_high/2_process_data_videvo_high_flow_2_50_num_21405.json"]

    # json_path = "/share/project/panting/share_datasets/raw_video_data/45x384x240/gaopin_high/"
    # json_files = ["/share/project/denghaoge/nova/project/preprocess_data/json/gaopin_1080p/2_process_data_gaopin_1080p_flow_2_50_num_248771.json"]

    json_path = "/share/project/panting/share_datasets/raw_video_data/45x384x240/gaopin_low/"
    json_files = ["/share/project/denghaoge/nova/project/preprocess_data/json/gaopin_1080p/2_process_data_gaopin_1080p_flow_2_50_num_248771.json"]

    # json_files = ["/share/project/denghaoge/Autoregressive/Dataset/hunyuan_video_data/skyworks_70k/1_transformed_data_huanyuan_video_prompt_skyworks_70k.json"]

    # json_files = ["/share/project/denghaoge/nova/project/preprocess_data/json/gaopin_600p/2_process_data_gaopin_600p_2_50_num_7754381.json"]
    # fmt: on

    volume_cnt, uid, rank_data_dict = 0, 0, []
    os.makedirs(json_path, exist_ok=True)
    for json_file in json_files:
        world_data_dict = load_and_split(json_file, **vid_args)
        np.random.shuffle(world_data_dict)
        for data_dict in world_data_dict:
            rank_data_dict.append(data_dict)
            rank_data_dict[-1]["id"] = str(uid).zfill(9)
            uid += 1
            if len(rank_data_dict) >= 1000:
                with open(json_path + "{}.json".format(str(volume_cnt).zfill(6)), "w") as f:
                    json.dump(rank_data_dict, f)
                rank_data_dict, volume_cnt = [], volume_cnt + 1
        print(json_file, "+", len(world_data_dict), "=", uid)
    if rank_data_dict:
        with open(json_path + "{}.json".format(str(volume_cnt).zfill(6)), "w") as f:
            json.dump(rank_data_dict, f)
