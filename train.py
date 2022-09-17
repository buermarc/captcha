from pathlib import Path
import os
import torch
from torch.utils.data import DataLoader
from captcha_dataset import CaptachDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from custom_rcnn_lightning_model import CustomRcnnLightningModel
from pytorch_lightning.loggers import TensorBoardLogger

PREFERRED_DATATYPE = torch.double
BATCH_SIZE = 2


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == '__main__':
    core_count = os.cpu_count()

    train_data = CaptachDataset(
        image_path=Path("generated_captchas"),
        label_file=Path("labels.json"),
        preferred_datatyp=PREFERRED_DATATYPE
    )
    val_data = CaptachDataset(
        image_path=Path("val_generated_captchas"),
        label_file=Path("val_labels.json"),
        preferred_datatyp=PREFERRED_DATATYPE
    )

    train_dataloader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=core_count if core_count else 4
    )

    val_dataloader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=core_count if core_count else 4
    )

    model = CustomRcnnLightningModel()

    logger = TensorBoardLogger("tb_logs", name="CustomRcnnLightningModel")

    early_stopping = EarlyStopping(
        monitor="val_loss_mean",
        min_delta=0.001,
        patience=5,
        mode="max",
        verbose=True,
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[early_stopping],
        max_epochs=100,
        logger=logger,
        log_every_n_steps=1,
    )
    trainer.fit(model.to(PREFERRED_DATATYPE), train_dataloader, val_dataloader)
