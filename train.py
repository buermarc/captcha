from pathlib import Path
import os
import torch
from torch.utils.data import DataLoader
from captcha_dataset import CaptachDataset
import pytorch_lightning as pl
from custom_rcnn_lightning_model import CustomRcnnLightningModel


def collate_fn(batch):
    return tuple(zip(*batch))


train_data = CaptachDataset(
    image_path=Path("generated_captchas"), label_file=Path("labels.json")
)


core_count = os.cpu_count()
train_dataloader = DataLoader(
    train_data,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=int(core_count / 2) if core_count else 4,
)

model = CustomRcnnLightningModel()

trainer = pl.Trainer(
    gpus=1 if torch.cuda.is_available() else 0,
    max_epochs=2,
)

trainer.fit(model.double(), train_dataloader)
