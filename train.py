from pathlib import Path
from torch.utils.data import DataLoader
from captcha_dataset import CaptachDataset
import pytorch_lightning as pl
from custom_rcnn_lightning_model import CustomRcnnLightningModel

train_data = CaptachDataset(
    image_path=Path("captchas"),
    label_file=Path("labels.json"),
)


def collate_fn(batch):
    return tuple(zip(*batch))

train_dataloader = DataLoader(train_data, batch_size=2, shuffle=True, collate_fn=collate_fn)

model = CustomRcnnLightningModel()

trainer = pl.Trainer(
    gpus=1,
    max_epochs=2,
)

trainer.fit(model, train_dataloader)
