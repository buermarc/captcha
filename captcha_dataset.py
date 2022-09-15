from pathlib import Path
from os.path import basename
import json
from torch.utils.data import Dataset
import torch
from typing import Tuple, Union, Dict, List, Any
from torchvision.io import read_image
import glob


class CaptachDataset(Dataset):

    def __init__(
        self,
        image_path: Path,
        label_file: Path,
        use_cache: bool = False,
        transform: Any = None,
        file_ending: str = "png"
    ):
        self.image_path = image_path
        self.label_file = label_file
        with open(self.label_file, mode="r") as _file:
            self.labels_json = json.load(_file)

        self.use_cache = use_cache
        self.transform = transform
        self.file_ending = file_ending

        self.images = glob.glob(f"{image_path}/*.{self.file_ending}")

        assert self.images, f"No images found in path {image_path}"

        if self.use_cache:
            # Use multiprocessing to load images and targets into RAM
            from multiprocessing import Pool
            with Pool() as pool:
                self.cached_data = pool.starmap(self._load_index, next(iter(range(len(self.images)))))

    def __len__(self) -> int:
        return len(self.images)

    def _load_index(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, List[Dict[str, Union[str, int]]]]:
        image = read_image(self.images[index])
        content = basename(self.images[index]).replace(f".{self.file_ending}", "")
        labels = self.labels_json[content]
        return image, labels

    def __getitem__(
        self,
        index: int
    ) -> Tuple[torch.Tensor, List[Dict[str, Union[str, int]]]]:

        if self.use_cache:
            image, labels = self.cached_data[index]
        else:
            image, labels = self._load_index(index)

        if self.transform:
            image = self.transform(image)

        return image, labels