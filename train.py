from pathlib import Path
import os
import torch
from torch.utils.data import DataLoader
from captcha_dataset import CaptachDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from custom_rcnn_lightning_model import CustomRcnnLightningModel


def collate_fn(batch):
    return tuple(zip(*batch))


core_count = os.cpu_count()

train_data = CaptachDataset(
    image_path=Path("generated_captchas"), label_file=Path("labels.json")
)
val_data = CaptachDataset(
    image_path=Path("val_generated_captchas"),
    label_file=Path("val_labels.json")
)

train_dataloader = DataLoader(
    train_data,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=int(core_count / 2) if core_count else 4,
)

val_dataloader = DataLoader(
    val_data,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=int(core_count / 2) if core_count else 4,
)

model = CustomRcnnLightningModel()

early_stopping = EarlyStopping(
    monitor="val_map",
    min_delta=0.001,
    patience=5,
)

trainer = pl.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    callbacks=[early_stopping],
    max_epochs=100,
)

trainer.fit(model.double(), train_dataloader, val_dataloader)
