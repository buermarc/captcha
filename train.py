from pathlib import Path
from torch.utils.data import DataLoader
from captcha_dataset import CaptachDataset

train_data = CaptachDataset(
    image_path=Path("captchas"),
    label_file=Path("labels.json"),
)

train_dataloader = DataLoader(train_data, batch_size=2, shuffle=True)


for x, y in train_dataloader:
    print(x, y)
