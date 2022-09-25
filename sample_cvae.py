import glob
import os
import string

import torch
from torchvision.utils import save_image

import utils
from cvae import CVAE

VERSION = 38
LATENT_DIM = 350

model_path = glob.glob(f"cvae_tb_logs/CVAE/version_{37}/models/*.ckpt")[0]
model = CVAE.load_from_checkpoint(
    model_path, latent_dim=LATENT_DIM, num_classes=62, channel_width_height=(3, 30, 60)
)

letters = string.ascii_letters + string.digits

try:
    os.mkdir("sample_cvae_images")
except Exception as exc:
    print(exc)
labels = torch.Tensor(utils.encode_label([str(letter) for letter in [*letters]]))
labels = labels.reshape(labels.shape[0], 1).to("cuda").to(torch.int64)

model.device_str = "cuda"
model.to("cuda")
result = model.sampling(n=62, c=labels)

for res, letter in zip(result, letters):
    with open(f"sample_cvae_images/{letter}.png", mode="wb+") as _file:
        save_image(res, _file)
