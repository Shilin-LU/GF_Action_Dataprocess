export TF_CPP_MIN_LOG_LEVEL=3
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_OFFLINE=1
export DS_ENV_FILE=/share/project/panting/configs/deepspeed/.deepspeed_env
export PYTHONPATH=/share/project/denghaoge/Tag_video/recognize-anything-main

# nohup deepspeed --no_local_rank --master_port 43337 \
# -H /share/project/panting/configs/deepspeed/deng16.hostfile \
# /share/project/panting/share_datasets/vid_devkit/scripts/tag_decord.py \
# --record /share/project/panting/share_datasets/raw_video_data/49x384x240/pexels_flow \
# --jsons /share/project/panting/share_datasets/raw_video_data/49x384x240/pexels \
# --model /share/project/denghaoge/Tag_video/Weight/recognize-anything-plus-model/ram_plus_swin_large_14m.pth > pexels.log 2>&1 &

# nohup deepspeed --no_local_rank --master_port 43338 \
# -H /share/project/panting/configs/deepspeed/deng16.hostfile \
# /share/project/panting/share_datasets/vid_devkit/scripts/tag_decord.py \
# --record /share/project/panting/share_datasets/raw_video_data/49x384x240/artgrid_flow \
# --jsons /share/project/panting/share_datasets/raw_video_data/49x384x240/artgrid \
# --model /share/project/denghaoge/Tag_video/Weight/recognize-anything-plus-model/ram_plus_swin_large_14m.pth > artgrid.log 2>&1 &

# nohup deepspeed --no_local_rank --master_port 43337 \
# -H /share/project/panting/configs/deepspeed/deng16.hostfile \
# /share/project/panting/share_datasets/vid_devkit/scripts/tag_decord.py \
# --record /share/project/panting/share_datasets/raw_video_data/49x384x240/skywork_youtube_tag \
# --jsons /share/project/panting/share_datasets/raw_video_data/49x384x240/skywork_youtube \
# --model /share/project/denghaoge/Tag_video/Weight/recognize-anything-plus-model/ram_plus_swin_large_14m.pth > skywork_youtube.log 2>&1 &

# nohup deepspeed --no_local_rank --master_port 43338 \
# -H /share/project/panting/configs/deepspeed/deng16.hostfile \
# /share/project/panting/share_datasets/vid_devkit/scripts/tag_decord.py \
# --record /share/project/panting/share_datasets/raw_video_data/49x384x240/skywork_human_flow \
# --jsons /share/project/panting/share_datasets/raw_video_data/49x384x240/skywork_human \
# --model /share/project/denghaoge/Tag_video/Weight/recognize-anything-plus-model/ram_plus_swin_large_14m.pth > skywork_human.log 2>&1 &

# nohup deepspeed --no_local_rank --master_port 43338 \
# -H /share/project/panting/configs/deepspeed/deng16.hostfile \
# /share/project/panting/share_datasets/vid_devkit/scripts/tag_decord.py \
# --record /share/project/panting/share_datasets/raw_video_data/49x768x480/gaopin_low_flow \
# --jsons /share/project/panting/share_datasets/raw_video_data/49x768x480/gaopin_low \
# --model /share/project/denghaoge/Tag_video/Weight/recognize-anything-plus-model/ram_plus_swin_large_14m.pth > gaopin_low2.log 2>&1 &

nohup deepspeed --no_local_rank --master_port 43339 \
-H /share/project/denghaoge/shilinlu/proj/simple_ursa/accelerate_configs/9_nodes_deepspeed.hostfile \
/share/project/panting/share_datasets/vid_devkit/scripts/tag_decord.py \
--record /share/project/denghaoge/shilinlu/dataset/GF-Minecraft/data_2003_tag \
--jsons /share/project/denghaoge/shilinlu/dataset/GF-Minecraft/data_2003/json \
--model /share/project/denghaoge/Tag_video/Weight/recognize-anything-plus-model/ram_plus_swin_large_14m.pth > GF2003_20251108.log 2>&1 &

# nohup deepspeed --no_local_rank \
# /share/project/panting/share_datasets/vid_devkit/scripts/tag_decord.py \
# --record /share/project/panting/share_datasets/raw_video_data/49x768x480/gaopin_low_flow \
# --jsons /share/project/panting/share_datasets/raw_video_data/49x768x480/gaopin_low \
# --model /share/project/denghaoge/Tag_video/Weight/recognize-anything-plus-model/ram_plus_swin_large_14m.pth \
# > skywork_youtube.log 2>&1 &
