import os
import numpy as np

root = "/share/project/panting/share_datasets/pt52_phi256_train/"
vid_files = [root + x for x in os.listdir(root) if x.endswith(".data")]
root = "/share/project/panting/share_datasets/hq52_phi256_train/"
vid_files += [root + x for x in os.listdir(root) if x.endswith(".data")]
vid_files.sort()

np.random.seed(1337)
np.random.shuffle(vid_files)

# to = "/share/project/panting/share_datasets/mix52_cog384_train/"
# for i, src1 in enumerate(vid_files):
#     tgt1 = to + str(i).zfill(6) + ".data"
#     src2, tgt2 = src1.replace(".data", ".index"), tgt1.replace(".data", ".index")
#     print(src1, tgt1)
#     os.symlink(src1, tgt1)
#     print(src2, tgt2)
#     os.symlink(src2, tgt2)

to = "/share/project/panting/share_datasets/mix52_phi256_train/"
for i, src1 in enumerate(vid_files):
    src1 = src1.replace("opensora480", "phi256")
    tgt1 = to + str(i).zfill(6) + ".data"
    src2, tgt2 = src1.replace(".data", ".index"), tgt1.replace(".data", ".index")
    print(src1, tgt1)
    os.symlink(src1, tgt1)
    print(src2, tgt2)
    os.symlink(src2, tgt2)
