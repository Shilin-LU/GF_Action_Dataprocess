import subprocess
import os

video = "/share/project/denghaoge/shilinlu/dataset/GF-Minecraft/data_269/video/seed_186_part_186.mp4"
out_dir = "frames"

os.makedirs(out_dir, exist_ok=True)

# 1. extract frames
subprocess.run([
    "ffmpeg", "-y",
    "-i", video,
    os.path.join(out_dir, "frame_%05d.png")
])

# 2. rebuild video at 16fps
subprocess.run([
    "ffmpeg", "-y",
    "-framerate", "16",
    "-i", os.path.join(out_dir, "frame_%05d.png"),
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-preset", "medium",
    "-crf", "18",
    "output_16fps.mp4"
])

print("Done!")
