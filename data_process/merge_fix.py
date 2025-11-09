import os
import shutil
import codewithgpu
import multiprocessing

path1 = "/share/project/panting/share_datasets/hq52_cog384_train/"
path2 = "/share/project/panting/share_datasets/hq52_cog384_train_fix/"
path3 = "/share/project/panting/share_datasets/hq52_cog384_train_final/"


def actor_fn(input_queue):
    while True:
        i = input_queue.get()
        if i is None:
            break
        record_path = path3 + str(i // num_volumes).zfill(6)
        dataset_path = path1 + str(i).zfill(6)
        if not os.path.exists(dataset_path):
            dataset_path = path2 + str(i).zfill(6)
        feats = codewithgpu.RecordDataset(dataset_path)._features
        writer = codewithgpu.RecordWriter(record_path, feats, zfill_width=6)
        for j in range(i, i + num_volumes):
            dataset_path = path1 + str(j).zfill(6)
            if not os.path.exists(dataset_path):
                dataset_path = path2 + str(j).zfill(6)
            dataset = codewithgpu.RecordDataset(dataset_path)
            for example in dataset:
                writer.write(example)
        writer.close()
        print("Done", record_path)


num_volumes, num_actors, q = 10, 128, multiprocessing.Queue()
actors = [multiprocessing.Process(target=actor_fn, args=(q,)) for _ in range(num_actors)]
for i in range(0, 10560, num_volumes):
    record_path = path3 + str(i // num_volumes).zfill(6)
    if os.path.exists(record_path + "/METADATA"):
        continue
    shutil.rmtree(record_path)
    os.makedirs(record_path), q.put(i)
[q.put(None) for actor in actors]
[actor.start() for actor in actors], [actor.join() for actor in actors]

# files1 = [path1 + x for x in os.listdir(path1)]
# files2 = [path2 + x for x in os.listdir(path2)]
# files2.sort()
# num_ok = 0
# ok_indices = set()
# for file in files2:
#     assert os.path.exists(file + "/METADATA")
#     ok_indices.add(int(os.path.splitext(os.path.split(file)[-1])[0]))
#     num_ok += 1
#     if os.path.exists(path1 + os.path.split(file)[-1]):
#         print("RM", path1 + os.path.split(file)[-1])
#         shutil.rmtree(path1 + os.path.split(file)[-1])
#     # assert not os.path.exists(path1 + os.path.split(file)[-1] + "/METADATA")
# for file in files1:
#     if os.path.exists(path1 + os.path.split(file)[-1] + "/METADATA"):
#         ok_indices.add(int(os.path.splitext(os.path.split(file)[-1])[0]))
#         num_ok += 1
# print(num_ok)
# for i in range(10560):
#     if i not in ok_indices:
#         print(i)
