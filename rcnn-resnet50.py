from PIL import Image
import torch, torchvision
from torchvision import datasets
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import utils


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
# Training


#output = model(images, targets)


# Evaluation
model.eval()

testdata = []
#data = Image.open("./data/sample/01.jpg")
#testdata.append(torchvision.transforms.ToTensor()(data))
data = Image.open("./data/sample/02.jpg")
testdata.append(torchvision.transforms.ToTensor()(data))

#testdataloader = torch.utils.data.DataLoader(testdata, batch_size=1, shuffle=False)

predictions = model(testdata)
utils.print_dataset(predictions[0])
#utils.print_dataset(predictions[1])
utils.show_dataset(data, predictions[0])

# optionally, if you want to export the model to ONNX:
#torch.onnx.export(model, x, "faster_rcnn.onnx", opset_version = 11)


