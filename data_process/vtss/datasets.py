# ------------------------------------------------------------------------
# Copyright (c) 2024-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------

import numpy as np
import decord


def get_frames_slow(video_path, start=0, end=None):
    reader = decord.VideoReader(video_path)
    fsize_t, fragments_t, frame_interval, resize = 32, 1, 2, 1280
    num_frames = end - start if end else len(reader)
    h, w = reader[0].shape[:2]
    scale = float(resize) / float(max(h, w))
    size = int(h * scale + 0.5), int(w * scale + 0.5)
    reader = decord.VideoReader(video_path, height=size[0], width=size[1])
    tgrids = np.array([num_frames // fragments_t * i for i in range(fragments_t)])
    tlength = num_frames // fragments_t
    if tlength > fsize_t * frame_interval:
        rnd_t = np.random.randint(0, tlength - fsize_t * frame_interval, size=len(tgrids))
    else:
        rnd_t = np.zeros(len(tgrids), dtype=np.int32)
    ranges_t = np.arange(fsize_t)[None, :] * frame_interval + rnd_t[:, None] + tgrids[:, None]
    ranges_t = np.concatenate(ranges_t) % num_frames + start
    return reader.get_batch(ranges_t).asnumpy()


def get_frames(video_path, start=0, end=64, stride=2):
    reader = decord.VideoReader(video_path)
    h, w = reader[0].shape[:2]
    scale = float(1280) / float(max(h, w))
    size = int(h * scale + 0.5), int(w * scale + 0.5)
    reader = decord.VideoReader(video_path, height=size[0], width=size[1])
    return reader.get_batch(range(start, end, stride)).asnumpy()


def get_spatial_fragments(
    video,
    fragments_h=7,
    fragments_w=7,
    fsize_h=32,
    fsize_w=32,
    aligned=32,
    random=False,
    **kwargs,
):
    video = video.transpose((3, 0, 1, 2))  # C, T, H, W
    size_h = fragments_h * fsize_h  # 224
    size_w = fragments_w * fsize_w  # 224
    if video.shape[1] == 1:
        aligned = 1
    dur_t, res_h, res_w = video.shape[-3:]
    size = size_h, size_w
    # fmt: off
    hgrids = [min(res_h // fragments_h * i, res_h - fsize_h) for i in range(fragments_h)]
    wgrids = [min(res_w // fragments_w * i, res_w - fsize_w) for i in range(fragments_w)]
    # fmt: on
    hlength, wlength = res_h // fragments_h, res_w // fragments_w
    grid_shape = (len(hgrids), len(wgrids), dur_t // aligned)
    if hlength > fsize_h:
        rnd_h = np.random.randint(0, hlength - fsize_h, grid_shape)
    else:
        rnd_h = np.zeros(grid_shape, dtype="int32")
    if wlength > fsize_w:
        rnd_w = np.random.randint(0, wlength - fsize_w, grid_shape)
    else:
        rnd_w = np.zeros(grid_shape, dtype="int32")
    target_video = np.zeros(video.shape[:-2] + size, dtype=video.dtype)
    for i, hs in enumerate(hgrids):
        for j, ws in enumerate(wgrids):
            for t in range(dur_t // aligned):
                t_s, t_e = t * aligned, (t + 1) * aligned
                h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                w_s, w_e = j * fsize_w, (j + 1) * fsize_w
                h_so, h_eo = hs + rnd_h[i][j][t], hs + rnd_h[i][j][t] + fsize_h
                w_so, w_eo = ws + rnd_w[i][j][t], ws + rnd_w[i][j][t] + fsize_w
                target_video[:, t_s:t_e, h_s:h_e, w_s:w_e] = video[:, t_s:t_e, h_so:h_eo, w_so:w_eo]
    return target_video
