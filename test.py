from starlette.responses import JSONResponse
import torchvision
from io import BytesIO
import utils
import torchvision, os, utils
from typing import Dict
import torch
from PIL import Image
from custom_rcnn_lightning_model import CustomRcnnLightningModel

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

pictures: Dict[str, Image.Image] = {}
outputs = []


def index() -> HTMLResponse:
    _html = "<html>\n"
    for key in pictures.keys():
        _html += f'<img src="/image/{key}" alt="/image/{key}">\n'
    _html += "</html>"

    return HTMLResponse(_html)


def serve_image_file(file_name: str) -> StreamingResponse:
    image = BytesIO()
    img = pictures[file_name]
    img.save(image, format='PNG', quality=85)
    image.seek(0)
    return StreamingResponse(image, media_type="image/png")


def load() -> None:
    version = 21
    checkpoint_file = os.listdir(f"./tb_logs/CustomRcnnLightningModel/version_{version}/checkpoints/")[0]

    model = CustomRcnnLightningModel.load_from_checkpoint(f"./tb_logs/CustomRcnnLightningModel/version_{version}/checkpoints/{checkpoint_file}")

    images = {}
    image_files = os.listdir("./data/test/")

    image_tensors = torch.zeros((len(image_files), 3, 60, 160))
    for idx, image in enumerate(image_files):
        testimage = Image.open(f"./data/test/{image}")
        images[image] = testimage
        image_tensors[idx] = torchvision.transforms.ToTensor()(testimage)

    outputs.extend(model.forward(image_tensors))

    threshold = 0.9

    for image_file, output in zip(image_files, outputs):
        image = images[image_file]
        pictures[image_file] = utils.show_dataset(image, output, ret=True, threshold=threshold)

    for output in outputs:
        utils.print_dataset(output)
    print(utils.own_testmetric([image.replace(".png", "") for image in images], outputs, threshold=threshold))


def create_app() -> FastAPI:
    app = FastAPI()
    load()
    app.add_api_route("/", index)
    app.add_api_route("/image/{file_name}", serve_image_file)
    return app
