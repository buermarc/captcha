import torchvision
import utils
from PIL import Image
from custom_rcnn_lightning_model import CustomRcnnLightningModel

model = CustomRcnnLightningModel.load_from_checkpoint("tb_logs/CustomRcnnLightningModel/version_2/checkpoints/epoch=2-step=210.ckpt")

testimage = Image.open("data/test/YQ2WV.png")
testtensor = [torchvision.transforms.ToTensor()(testimage)]

output = model.forward(testtensor)

utils.show_dataset(testimage, output[0], threshold=0.1)