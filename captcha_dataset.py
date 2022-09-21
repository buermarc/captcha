import glob
import json
from os.path import basename
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

import utils


class CaptachDataset(Dataset):
    def __init__(
        self,
        image_path: Path,
        label_file: Path,
        use_cache: bool = False,
        transform: Any = None,
        file_ending: str = "png",
        preferred_datatyp=torch.double,
    ):
        self.image_path = image_path
        self.label_file = label_file
        with open(self.label_file, mode="r") as _file:
            labels_json = json.load(_file)
            self.labels_json = {
                content: {
                    "boxes": torch.Tensor(labels_json[content]["boxes"]).to(
                        torch.int64
                    ),
                    "labels": torch.Tensor(
                        utils.encode_label(labels_json[content]["labels"])
                    ).to(torch.int64),
                }
                for content in labels_json.keys()
            }

        self.use_cache = use_cache
        self.transform = transform
        self.file_ending = file_ending
        self.preferred_datatype = preferred_datatyp

        self.images = glob.glob(f"{image_path}/*.{self.file_ending}")

        assert self.images, f"No images found in path {image_path}"

        if self.use_cache:
            # Use multiprocessing to load images and targets into RAM
            from multiprocessing import Pool

            with Pool() as pool:
                self.cached_data = pool.starmap(self._load_index, iter(self.images))

    def __len__(self) -> int:
        return len(self.images)

    def _load_index(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image = read_image(self.images[index]).to(self.preferred_datatype) / 255
        content = basename(self.images[index]).replace(f".{self.file_ending}", "")
        labels = self.labels_json[content]

        return image, labels

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        if self.use_cache:
            image, labels = self.cached_data[index]
        else:
            image, labels = self._load_index(index)

        if self.transform:
            image = self.transform(image)

        return image, labels
