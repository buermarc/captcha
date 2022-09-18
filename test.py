import torchvision
import utils
import torchvision, os, utils
from PIL import Image
from custom_rcnn_lightning_model import CustomRcnnLightningModel


version = 14
checkpoint_file = os.listdir(f"./tb_logs/CustomRcnnLightningModel/version_{version}/checkpoints/")[0]

model = CustomRcnnLightningModel.load_from_checkpoint(f"./tb_logs/CustomRcnnLightningModel/version_{version}/checkpoints/{checkpoint_file}")

outputs = []
for testimage_file in os.listdir("./data/test/"):
    testimage = Image.open(f"./data/test/{testimage_file}")
    testtensor = [torchvision.transforms.ToTensor()(testimage)]

    output = model.forward(testtensor)
    outputs.append[output]
    #utils.show_dataset(testimage, output[0], threshold=0.1)

utils.own_testmetric(testimage_file.replace(".png", ""), outputs, threshold=0)
