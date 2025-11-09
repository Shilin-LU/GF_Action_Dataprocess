import os
import json
import cv2
import sys
import pathlib
from multiprocessing import Pool, cpu_count
import tqdm

root = "/share/project/panting/share_datasets/vid_devkit/panda6m/"
video_root = "/share/project/lzx/video_data/ks3_extracted_videos/panda70m_sora_slected_6m"


def get_video_info(video_path, return_codec=False):
    if not os.path.exists(video_path):
        return "unknown", 0
    video, codec = cv2.VideoCapture(video_path), "unknown"
    if return_codec:
        codec = video.get(cv2.CAP_PROP_FOURCC)
        codec = int(codec).to_bytes(4, byteorder=sys.byteorder).decode()
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    video.release()
    return codec, frame_count


def process(data_dict):
    name, codec0 = pathlib.Path(data_dict["path"]).stem, None
    frame_counts = []
    for i, (st, ed) in enumerate(data_dict["segments"]):
        video_name = f"{video_root}/{name}_{st}_{ed}.mp4"
        codec, frame_count = get_video_info(video_name, i == 0)
        codec0 = codec if i == 0 else codec0
        frame_counts.append(frame_count)
    return {"codec": codec0, "frame_counts": frame_counts, "idx": data_dict["idx"]}


json_files = [root + x for x in os.listdir(root)]
json_files.sort()
json_files = [json_files[9]]


for json_file in json_files:
    list_data_dict = json.load(open(json_file))
    num_cores = min(cpu_count(), 256)
    inputs = []
    for i, data_dict in enumerate(list_data_dict):
        segments = []
        for cap_dict in [data_dict["captions"][i] for i in data_dict["segments"]]:
            segments.append([cap_dict["start"], cap_dict["end"]])
        inputs.append({"path": data_dict["path"], "segments": segments, "idx": i})
    with Pool(num_cores) as pool:
        outputs = list(tqdm.tqdm(pool.imap(process, inputs), total=len(inputs)))
    out_data_dict = []
    for output in outputs:
        data_dict = list_data_dict[output["idx"]]
        data_dict["video_info"]["codec"] = output["codec"]
        keep_cap_dict = [data_dict["captions"][i] for i in data_dict["segments"]]
        assert len(keep_cap_dict) == len(output["frame_counts"])
        for cap_dict, frame_count in zip(keep_cap_dict, output["frame_counts"]):
            cap_dict["frame_count"] = frame_count
        data_dict["captions"] = keep_cap_dict
        out_data_dict.append(data_dict)
    print(json_file)
    with open(json_file.replace("panda6m", "panda6m_codec"), "w") as f:
        json.dump(out_data_dict, f)


# # fmt: off
# num_json_files = 10
# with open("/share/project/panting/share_datasets/vid_devkit/raw/panda_hq6m.json") as f:
#     list_data_dict = json.load(f)
#     num_per_files = len(list_data_dict) // num_json_files + 1
#     print(num_per_files)
#     for rank in range(num_json_files):
#         a = list_data_dict[rank * num_per_files : (rank + 1) * num_per_files]
#         print(rank * num_per_files, (rank + 1) * num_per_files)
#         with open("/share/project/panting/share_datasets/vid_devkit/panda6m/{}.json".format(str(rank).zfill(4)), "w") as f:
#             json.dump(a, f)
# # fmt on
