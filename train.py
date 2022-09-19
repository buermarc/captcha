from pathlib import Path
import os
import torch
from torch.utils.data import DataLoader
from captcha_dataset import CaptachDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from custom_rcnn_lightning_model import CustomRcnnLightningModel
from pytorch_lightning.loggers import TensorBoardLogger
from torchinfo import summary

PREFERRED_DATATYPE = torch.double
BATCH_SIZE = 1
DATADIR = "data/"


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == '__main__':
    core_count = os.cpu_count()

    train_data = CaptachDataset(
        image_path=Path(DATADIR + "train"),
        label_file=Path(DATADIR + "train_labels.json"),
        preferred_datatyp=PREFERRED_DATATYPE
    )
    val_data = CaptachDataset(
        image_path=Path(DATADIR + "val"),
        label_file=Path(DATADIR + "val_labels.json"),
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

    model = CustomRcnnLightningModel(num_classes=62+1)
    summary(model, device='cpu', input_size=(BATCH_SIZE, 3, 60, 160))

    logger = TensorBoardLogger("tb_logs", name="CustomRcnnLightningModel")
    logger.log_hyperparams({
        "batch_size": BATCH_SIZE,
        "train_data_size": len(train_data),
        "val_data_size": len(val_data),
    })

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
        log_every_n_steps=50,
    )
    trainer.fit(model.to(PREFERRED_DATATYPE), train_dataloader, val_dataloader)
