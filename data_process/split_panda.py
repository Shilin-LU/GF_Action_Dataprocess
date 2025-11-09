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
import json
import pathlib

# error_set = json.load(open("hq_errors.json"))
# error_set = set(error_set["vfr_error"] + error_set["frame_error"])
error_set = set()
# print(len(error_set))
video_root = "/share/project/lzx/video_data/ks3_extracted_videos/panda70m_sora_slected_6m"


def load_and_split(
    json_file, seqlen=29, fps_range=(23.5, 30.5), stride=2, shape=(480, 640), codec="FMP4"
):
    """Load video json annotations."""
    min_frame_count = (seqlen - 1) * stride + 1
    merge_data_dict, split_data_dict = json.load(open(json_file)), []
    base_size = shape[0]
    for data_dict in merge_data_dict:
        info = data_dict["video_info"]
        if info["codec"] != codec:
            continue
        if info["fps"] < fps_range[0] or info["fps"] > fps_range[1]:
            continue
        scale = base_size / min(info["shape"])
        if any(int(x * scale + 0.5) < y for (x, y) in zip(info["shape"], shape)):
            continue
        name = pathlib.Path(data_dict["path"]).stem
        assert len(data_dict["captions"]) == len(data_dict["segments"])
        for cap_dict in data_dict["captions"]:
            st, ed = cap_dict["start"], cap_dict["end"]
            video_name = f"{video_root}/{name}_{st}_{ed}.mp4"
            st, ed = (0, cap_dict["frame_count"])
            if (ed - st) < min_frame_count:
                continue
            center_st = max((ed - st - min_frame_count), 0) // 2
            center_ed = min(center_st + min_frame_count, ed)
            assert center_ed - center_st == min_frame_count, (center_st, center_ed)
            cap_data_dict = {"start": center_st, "end": center_ed, "video": video_name}
            cap_data_dict.update({"text": "", "caption": cap_dict["caption"]})
            split_data_dict += [{"flow": 5.0, **cap_data_dict}]
    return split_data_dict


if __name__ == "__main__":
    # fmt: off
    # hq6m = json.load(open("/share/project/panting/share_datasets/vid_devkit/scripts/panda_hq6m_with_info.json"))
    # print(hq6m[0])
    # while True: pass
    # for data_dict in hq6m:
    #     data_dict["video_info"] = hq6m_info[data_dict["video_id"]]
    #     num += len(data_dict["segments"])
    # print(len(hq6m), num)
    # with open("/share/project/panting/share_datasets/vid_devkit/scripts/panda_hq6m_with_info.json", "w") as f:
    #     json.dump(hq6m, f)
    root = "/share/project/panting/share_datasets/vid_devkit/panda6m_codec/"
    json_files = [root + x for x in os.listdir(root)]
    json_files.sort()

    # hq6m_index = json.load(open("/share/project/panting/share_datasets/vid_devkit/raw/panda_hq6m_index.json"))
    # hq6m_set = set(hq6m_index.keys())
    # f = open("/share/project/panting/share_datasets/vid_devkit/raw/panda70m.jsonl", "r")
    # list_data_dict = []
    # for i, data in enumerate(f.readlines()):
    #     if i % 1000000 == 0:
    #         print("Process", i)
    #     data = json.loads(data)
    #     if data["video_id"] not in hq6m_set:
    #         continue
    #     data["segments"] = hq6m_index[data["video_id"]]
    #     list_data_dict.append(data)
    #     if len(list_data_dict) % 10000 == 0:
    #         print("Get", len(list_data_dict))
    # with open("panda_hq6m.json", "w") as f:
    #     json.dump(list_data_dict, f)

    volume_cnt, uid, rank_data_dict = 0, 0, []

    volume_cnt, uid, fps_range, stride, codec = 0, 0, (23.5, 30.5), 2, "FMP4"
    json_path = "/share/project/panting/share_datasets/vid_devkit/data/captions_panda29_480p_stride2_fmp4"

    volume_cnt, uid, fps_range, stride, codec = 3748, 3747508, (23.5, 30.5), 2, "h264"
    json_path = "/share/project/panting/share_datasets/vid_devkit/data/captions_panda29_480p_stride2_h264"

    # fmt: on
    os.makedirs(json_path) if not os.path.exists(json_path) else None

    for json_file in json_files:
        world_data_dict = load_and_split(json_file, fps_range=fps_range, stride=stride, codec=codec)
        for data_dict in world_data_dict:
            rank_data_dict.append(data_dict)
            rank_data_dict[-1]["id"] = str(uid).zfill(9)
            uid += 1
            if len(rank_data_dict) >= 1000:
                print(json_path + "/{}.json".format(str(volume_cnt).zfill(6)))
                with open(json_path + "/{}.json".format(str(volume_cnt).zfill(6)), "w") as f:
                    json.dump(rank_data_dict, f)
                rank_data_dict, volume_cnt = [], volume_cnt + 1
        print(uid, json_file, len(world_data_dict))
    if rank_data_dict:
        print(json_path + "/{}.json".format(str(volume_cnt).zfill(6)))
        with open(json_path + "/{}.json".format(str(volume_cnt).zfill(6)), "w") as f:
            json.dump(rank_data_dict, f)
