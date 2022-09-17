import torchvision, torch, utils
from PIL import Image
from custom_rcnn_lightning_model import CustomRcnnLightningModel

model = CustomRcnnLightningModel.load_from_checkpoint("tb_logs/CustomRcnnLightningModel/version_4/checkpoints/epoch=2-step=24.ckpt")

testimage = Image.open("data/test/YQ2WV.png")
testtensor = [torchvision.transforms.ToTensor()(testimage)]

output = model.forward(testtensor)

utils.show_dataset(testimage, output[0])