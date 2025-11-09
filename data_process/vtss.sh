export TF_CPP_MIN_LOG_LEVEL=3
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_OFFLINE=1
export DS_ENV_FILE=/share/project/panting/configs/deepspeed/.deepspeed_env
export PYTHONPATH=/share/project/panting/share_models/recognize-anything-main

# nohup deepspeed --no_local_rank --master_port 43337 \
# -H /share/project/panting/configs/deepspeed/deng16.hostfile \
# /share/project/panting/share_datasets/vid_devkit/scripts/vtss_decord.py \
# --record /share/project/panting/share_datasets/raw_video_data/49x384x240/pexels_flow \
# --jsons /share/project/panting/share_datasets/raw_video_data/49x384x240/pexels \
# --model /share/project/denghaoge/Tag_video/Weight/recognize-anything-plus-model/ram_plus_swin_large_14m.pth > pexels.log 2>&1 &

# nohup deepspeed --no_local_rank --master_port 43338 \
# -H /share/project/panting/configs/deepspeed/deng16.hostfile \
# /share/project/panting/share_datasets/vid_devkit/scripts/vtss_decord.py \
# --record /share/project/panting/share_datasets/raw_video_data/49x384x240/artgrid_vtss \
# --jsons /share/project/panting/share_datasets/raw_video_data/49x384x240/artgrid \
# --model /share/project/denghaoge/Tag_video/Weight/recognize-anything-plus-model/ram_plus_swin_large_14m.pth > artgrid.log 2>&1 &

# nohup deepspeed --no_local_rank --master_port 43337 \
# -H /share/project/panting/configs/deepspeed/deng16.hostfile \
# /share/project/panting/share_datasets/vid_devkit/scripts/vtss_decord.py \
# --record /share/project/panting/share_datasets/raw_video_data/49x384x240/skywork_youtube_tag \
# --jsons /share/project/panting/share_datasets/raw_video_data/49x384x240/skywork_youtube \
# --model /share/project/denghaoge/Tag_video/Weight/recognize-anything-plus-model/ram_plus_swin_large_14m.pth > skywork_youtube.log 2>&1 &

# nohup deepspeed --no_local_rank --master_port 43338 \
# -H /share/project/panting/configs/deepspeed/deng16.hostfile \
# /share/project/panting/share_datasets/vid_devkit/scripts/vtss_decord.py \
# --record /share/project/panting/share_datasets/raw_video_data/49x384x240/skywork_human_flow \
# --jsons /share/project/panting/share_datasets/raw_video_data/49x384x240/skywork_human \
# --model /share/project/denghaoge/Tag_video/Weight/recognize-anything-plus-model/ram_plus_swin_large_14m.pth > skywork_human.log 2>&1 &

nohup deepspeed --no_local_rank --master_port 43338 \
-H /share/project/panting/configs/deepspeed/deng16.hostfile \
/share/project/panting/share_datasets/vid_devkit/scripts/vtss_decord.py \
--record /share/project/panting/share_datasets/raw_video_data/49x768x480/food_1086198_vtss \
--jsons /share/project/panting/share_datasets/raw_video_data/49x768x480/food_1086198 \
--model /share/project/denghaoge/Tag_video/Weight/recognize-anything-plus-model/ram_plus_swin_large_14m.pth > food.log 2>&1 &

# deepspeed --no_local_rank \
# /share/project/panting/share_datasets/vid_devkit/scripts/vtss_decord.py \
# --record /share/project/panting/share_datasets/raw_video_data/49x768x480/gaopin_low_vtss \
# --jsons /share/project/panting/share_datasets/raw_video_data/49x768x480/gaopin_low_tag \
# --model /share/project/denghaoge/Tag_video/Weight/recognize-anything-plus-model/ram_plus_swin_large_14m.pth 
