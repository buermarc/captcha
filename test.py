import torchvision
import utils
import torchvision, os, utils
import torch
from PIL import Image
from custom_rcnn_lightning_model import CustomRcnnLightningModel


version = 14
checkpoint_file = os.listdir(f"./tb_logs/CustomRcnnLightningModel/version_{version}/checkpoints/")[0]

model = CustomRcnnLightningModel.load_from_checkpoint(f"./tb_logs/CustomRcnnLightningModel/version_{version}/checkpoints/{checkpoint_file}")

images = os.listdir("./data/test/")

image_tensors = torch.zeros((len(images), 3, 60, 160))
for idx, image in enumerate(images):
    testimage = Image.open(f"./data/test/{image}")
    image_tensors[idx] = torchvision.transforms.ToTensor()(testimage)

outputs = model.forward(image_tensors)

print(utils.own_testmetric([image.replace(".png", "") for image in images], outputs, threshold=0.9))
