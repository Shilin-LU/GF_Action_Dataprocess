export TF_CPP_MIN_LOG_LEVEL=3
export DS_ENV_FILE=/share/project/panting/configs/deepspeed/.deepspeed_env
export PYTHONPATH=/share/project/panting/share_models/simple_ursa

# nohup deepspeed --no_local_rank --master_port 43337 \
# -H /share/project/panting/configs/deepspeed/deng16.hostfile \
# /share/project/panting/share_datasets/vid_devkit/scripts/cache_decord320.py \
# --record /share/project/panting/share_datasets/emu3_cosmos320/pexels \
# --jsons /share/project/panting/share_datasets/raw_video_data/49x384x240/pexels_tag \
# --vae /share/project/panting/mm_ckpt/vq/cosmos_4x8x8 > pexels.log 2>&1 &

# nohup deepspeed --no_local_rank --master_port 43338 \
# -H /share/project/panting/configs/deepspeed/deng16.hostfile \
# /share/project/panting/share_datasets/vid_devkit/scripts/cache_decord320.py \
# --record /share/project/panting/share_datasets/emu3_cosmos320/artgrid \
# --jsons /share/project/panting/share_datasets/raw_video_data/49x384x240/artgrid_tag \
# --vae /share/project/panting/mm_ckpt/vq/cosmos_4x8x8 > artgrid2.log 2>&1 &

# nohup deepspeed --no_local_rank --master_port 43337 \
# -H /share/project/panting/configs/deepspeed/deng16.hostfile \
# /share/project/panting/share_datasets/vid_devkit/scripts/cache_decord320.py \
# --record /share/project/panting/share_datasets/emu3_cosmos320/skywork_youtube \
# --jsons /share/project/panting/share_datasets/raw_video_data/49x384x240/skywork_youtube \
# --vae /share/project/panting/mm_ckpt/vq/cosmos_4x8x8 > skywork_youtube.log 2>&1 &

nohup deepspeed --no_local_rank --master_port 27862 \
-H /share/project/denghaoge/shilinlu/proj/simple_ursa/accelerate_configs/shilin_10node.hostfile \
/share/project/denghaoge/shilinlu/dataset/GF-Minecraft/data_process/cache_decord320.py \
--record /share/project/denghaoge/shilinlu/dataset/GF-Minecraft/GF_flow10_cache \
--jsons /share/project/denghaoge/shilinlu/dataset/GF-Minecraft/data_flow10_tag \
--vae /share/project/panting/mm_ckpt/vq/cosmos_4x8x8 > GF_flow10_cache.log 2>&1 &

# nohup deepspeed --no_local_rank --master_port 43338 \
# -H /share/project/panting/configs/deepspeed/deng16.hostfile \
# /share/project/panting/share_datasets/vid_devkit/scripts/cache_decord320.py \
# --record /share/project/panting/share_datasets/emu3_cosmos320/gaopin_low \
# --start 0 --end 10240 \
# --jsons /share/project/panting/share_datasets/raw_video_data/49x768x480/gaopin_low_tag \
# --vae /share/project/panting/mm_ckpt/vq/cosmos_4x8x8 > gaopin_low_2.log 2>&1 &

# nohup deepspeed --no_local_rank --master_port 43339 \
# -H /share/project/panting/configs/deepspeed/deng16.hostfile \
# /share/project/panting/share_datasets/vid_devkit/scripts/cache_decord320.py \
# --record /share/project/panting/share_datasets/emu3_cosmos320/koala_20250823 \
# --jsons /share/project/panting/share_datasets/raw_video_data/49x384x240/koala_20250823_tag \
# --vae /share/project/panting/mm_ckpt/vq/cosmos_4x8x8 > koala_20250823.log 2>&1 &

# deepspeed --no_local_rank \
# /share/project/panting/share_datasets/vid_devkit/scripts/cache_decord320.py \
# --record /share/project/panting/share_datasets/emu3_cosmos320/gaopin_low \
# --start 0 --end 10240 \
# --jsons /share/project/panting/share_datasets/raw_video_data/49x768x480/gaopin_low_tag \
# --vae /share/project/panting/mm_ckpt/vq/cosmos_4x8x8 
