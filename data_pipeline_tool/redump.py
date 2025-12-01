import codewithgpu
import os

dataset_src = "/share/project/denghaoge/shilinlu/dataset/GF-Minecraft/GF_training_r18_flow10"
dataset_tgt = "/share/project/denghaoge/shilinlu/dataset/GF-Minecraft/GF_training_r72_flow10"
os.makedirs(dataset_tgt, exist_ok=True)
max_examples = 17554 // 72 + 1

# 获取所有子目录（排除METADATA文件）
subdirs = []
for item in os.listdir(dataset_src):
    item_path = os.path.join(dataset_src, item)
    if os.path.isdir(item_path):
        subdirs.append(item_path)
subdirs.sort()

if not subdirs:
    raise ValueError(f"在 {dataset_src} 中未找到子目录")

# 从第一个子目录获取 features
first_dataset = codewithgpu.RecordDataset(subdirs[0])
features = first_dataset._features

# 创建 writer
writer = codewithgpu.RecordWriter(dataset_tgt, features=features, max_examples=max_examples, zfill_width=6)

# 遍历所有子目录并读取数据
for subdir in subdirs:
    dataset = codewithgpu.RecordDataset(subdir)
    for example in dataset:
        writer.write(example)

writer.close()