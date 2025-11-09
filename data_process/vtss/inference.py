import torch
import numpy as np

from vtss.model import DiViDeAddEvaluator
from vtss.datasets import get_frames, get_frames_v2, get_frames_v3, get_spatial_fragments


def inference_set(model, device, dtype=torch.float32):
    vids = [
        "/share/project/panting/share_datasets/vid_devkit/scripts/vtss/video/-KuN4XkUQBI_1.mp4",
        "/share/project/panting/share_datasets/vid_devkit/scripts/vtss/video/0psDrM3mxx0_155.mp4",
        "/share/project/panting/share_datasets/vid_devkit/scripts/vtss/video/J3JqVnOGfYA_3.mp4",
        "/share/project/panting/share_datasets/vid_devkit/scripts/vtss/video/RIeEULN2r2U_18.mp4",
        "/share/project/panting/share_datasets/vid_devkit/scripts/vtss/video/rKn-vWDMkwQ_33.mp4",
    ]
    # x = np.stack([get_spatial_fragments(get_frames(vid)) for vid in vids])
    # x = np.stack([get_frames_v2(vid) for vid in vids])
    x = np.stack([get_spatial_fragments(get_frames_v3(vid)) for vid in vids])

    x = torch.as_tensor(x).to(device).to(dtype)
    stdv = torch.tensor([58.395, 57.12, 57.375]).to(x)
    mean = torch.tensor([123.675, 116.28, 103.53]).to(x)
    x = x.transpose(1, -1).sub(mean).div_(stdv).transpose(1, -1)
    x, bsz = x.unflatten(2, (-1, 32)).permute(0, 2, 1, 3, 4, 5).flatten(0, 1), x.shape[0]
    y = model({"fragments": x}, reduce_scores=False)[0].unflatten(0, (bsz, -1))
    print(y.shape)
    return y.flatten(1).float().mean(1).tolist()


def main():
    device, dtype = torch.device("cuda", 0), torch.float16
    model = DiViDeAddEvaluator().to(device).to(dtype)
    ckpt_path = "/share/project/panting/share_datasets/vid_devkit/scripts/vtss/infer.pth"
    state_dict = torch.load(ckpt_path, weights_only=False)["state_dict"]
    model.load_state_dict(state_dict)
    torch.manual_seed(2337), np.random.seed(2337)
    print(inference_set(model, device, dtype))


if __name__ == "__main__":
    main()
