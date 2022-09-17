import torchvision
import utils
import torchvision, os, utils
from PIL import Image
from custom_rcnn_lightning_model import CustomRcnnLightningModel


version = 2
checkpoint_file = os.listdir(f"./tb_logs/CustomRcnnLightningModel/version_{version}/checkpoints/")[0]

model = CustomRcnnLightningModel.load_from_checkpoint(f"./tb_logs/CustomRcnnLightningModel/version_{version}/checkpoints/{checkpoint_file}")

testimage_file = os.listdir(f"./data/test/")[0]
testimage = Image.open(f"./data/test/{testimage_file}")
testtensor = [torchvision.transforms.ToTensor()(testimage)]

output = model.forward(testtensor)

utils.show_dataset(testimage, output[0], threshold=0.1)