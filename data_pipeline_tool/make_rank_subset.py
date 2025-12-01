import os
import numpy as np
import shutil
import json
import collections


def make_dataset(data_files, data_ranks, metadata):
    rank_data_files = collections.defaultdict(list)
    for i in range(0, len(data_files), len(data_ranks)):
        for j, data_file in zip(data_ranks, data_files[i : i + len(data_ranks)]):
            index_file = data_file.replace(".data", ".index")
            os.makedirs(out + str(j).zfill(3), exist_ok=True)
            rank_data_files[j].append(data_file)
    for rank in data_ranks:
        rank_dir = out + str(rank).zfill(3) + "/"
        index_files = []
        for i, j in enumerate(np.random.permutation(len(rank_data_files[rank]))):
            data_file = rank_data_files[rank][j]
            index_file = data_file.replace(".data", ".index")
            index_files.append(index_file)
            data_name = rank_dir + str(i).zfill(6)
            os.symlink(data_file, f"{data_name}.data"), os.symlink(index_file, f"{data_name}.index")
        metadata["entries"] = sum([len(open(index_file).readlines()) for index_file in index_files])
        json.dump(metadata, open(rank_dir + "METADATA", "w"))


vid_datasets = [
    "/share/project/denghaoge/shilinlu/dataset/GF-Minecraft/GFflow10cacher72_17554"  # TODO: 替换成你的第1个视频数据集路径
    # "/share/project/denghaoge/shilinlu/dataset/GF-Minecraft/GF269cache_20175",  # TODO: 替换成你的第2个视频数据集路径
]

img_datasets = []  # 没有图像数据

# 数据统计(如果目录名不包含数量,这行可能会报错,可以注释掉)
vid_size = sum(
    [
        int(os.path.basename(x.rstrip("/")).split("_")[-1])
        for x in vid_datasets
    ],
    0,
)
print("VID={}".format(vid_size))

ranks, img_devices = list(range(72)), 0  # 72个GPU全部用于视频训练
vid_ranks, img_ranks = ranks, []  # 所有ranks分配给视频,图像为空
out = "/share/project/denghaoge/shilinlu/dataset/GF-Minecraft/GF_training_r72_flow10/"  # TODO: 修改为你的输出目录路径

np.random.seed(1234)
vid_files = sum(
    [[os.path.join(d, x) for x in os.listdir(d) if x.endswith(".data")] for d in vid_datasets],
    [],
)
img_files = sum(
    [[os.path.join(d, x) for x in os.listdir(d) if x.endswith(".data")] for d in img_datasets],
    [],
)
vid_files.sort(), img_files.sort()
shutil.rmtree(out, ignore_errors=True)
os.makedirs(out)
make_dataset(vid_files, vid_ranks, json.load(open("METADATA")))
# 没有图像数据,注释掉图像处理
# make_dataset(img_files, img_ranks, json.load(open("METADATA2")))
