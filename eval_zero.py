import torchvision
import numpy as np
from tqdm import tqdm
import utils
import gc
import torchvision, os, utils
import torch
from PIL import Image
from custom_rcnn_lightning_model import CustomRcnnLightningModel
import itertools


version = 21
checkpoint_file = os.listdir(f"./tb_logs/CustomRcnnLightningModel/version_{version}/checkpoints/")[0]

model = CustomRcnnLightningModel.load_from_checkpoint(f"./tb_logs/CustomRcnnLightningModel/version_{version}/checkpoints/{checkpoint_file}")

def main():
    labels = []
    image_files = os.listdir("./data/test/")

    step_size = 8
    uncut_amount = len(image_files)/step_size
    amount = int(uncut_amount)

    for i in tqdm(range(amount)):
        subset = image_files[i*step_size:i*step_size+step_size]

        image_tensors = torch.zeros((len(subset), 3, 60, 160))
        for idx, image in enumerate(subset):
            testimage = Image.open(f"./data/test/{image}")
            image_tensors[idx] = torchvision.transforms.ToTensor()(testimage)
            del testimage
        outs = model.forward(image_tensors)
        labels.extend([out["labels"].tolist() for out in outs])
        del outs
        del image_tensors
        gc.collect()

    if uncut_amount != amount:
        subset = image_files[amount*step_size:]

        image_tensors = torch.zeros((len(subset), 3, 60, 160))
        for idx, image in enumerate(subset):
            testimage = Image.open(f"./data/test/{image}")
            image_tensors[idx] = torchvision.transforms.ToTensor()(testimage)
            del testimage
        labels.extend([out["labels"].tolist() for out in model.forward(image_tensors)])
        del image_tensors
        gc.collect()

    flatten = np.array([num for elem in labels for num in elem])
    breakpoint()
    uni, count = np.unique(flatten, return_counts=True)
    print(uni)
    print(count)


main()
