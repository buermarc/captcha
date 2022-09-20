import glob
import json
import os
import re
from io import BytesIO
from typing import Dict, Optional

import torch
import torchvision
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from PIL import Image

import utils
from custom_rcnn_lightning_model import CustomRcnnLightningModel

pictures: Dict[str, Optional[Image.Image]] = {}
outputs = []

VERSION = 51
THRESHOLD = 0.8


def index() -> HTMLResponse:
    _re = re.compile(r".*epoch_(?P<epoch>[0-9]+)_test_output.*-(?P<image>.+\.png)")
    print(pictures)
    _html = "<html>\n"
    keys = pictures.keys()
    epochs = [int(_re.match(key).group("epoch")) for key in keys]
    epochs.sort()
    images = set([_re.match(key).group("image") for key in keys])
    images = [str(image) for image in images]
    images.sort()
    _min = min(epochs)
    _max = max(epochs)

    _html = "<html>"
    for i in range(_min, _max + 1):
        for image in images:
            image_key = f"epoch_{i}_test_output-{image}"
            _html += f'<img src="/image/{image_key}" alt="/image/{image_key}">\n'
        _html += "<br>\n"
    _html += "</html>"

    return HTMLResponse(_html)


def serve_image_file(file_name: str) -> StreamingResponse:
    image = BytesIO()
    img = pictures[file_name]
    img.save(image, format="PNG", quality=85)
    image.seek(0)
    return StreamingResponse(image, media_type="image/png")


def load_v1() -> None:
    checkpoint_file = os.listdir(
        f"./tb_logs/CustomRcnnLightningModel/version_{VERSION}/checkpoints/"
    )[0]

    model = CustomRcnnLightningModel.load_from_checkpoint(
        f"./tb_logs/CustomRcnnLightningModel/version_{VERSION}/checkpoints/{checkpoint_file}"
    )

    images = {}
    image_files = os.listdir("./data/save_test/")

    image_tensors = torch.zeros((len(image_files), 3, 60, 160))
    for idx, image in enumerate(image_files):
        testimage = Image.open(f"./data/save_test/{image}")
        images[image] = testimage
        image_tensors[idx] = torchvision.transforms.ToTensor()(testimage)

    outputs.extend(model.forward(image_tensors))

    for image_file, output in zip(image_files, outputs):
        image = images[image_file]
        pictures[image_file] = utils.show_dataset(
            image, output, ret=True, threshold=THRESHOLD
        )

    for output in outputs:
        utils.print_dataset(output)
    print(
        utils.own_testmetric(
            [image.replace(".png", "") for image in images],
            outputs,
            threshold=THRESHOLD,
        )
    )


def load_v2() -> None:
    # load images from save_test dir
    # load outputs from current logdir
    # display images

    images = {}
    image_files = os.listdir("./data/save_test/")

    image_tensors = torch.zeros((len(image_files), 3, 60, 160))
    for idx, image in enumerate(image_files):
        testimage = Image.open(f"./data/save_test/{image}")
        images[image] = testimage
        image_tensors[idx] = torchvision.transforms.ToTensor()(testimage)

    output_files = glob.glob(
        f"./tb_logs/CustomRcnnLightningModel/version_{VERSION}/*.json"
    )

    output_versions = [
        json.load(open(output_file, mode="r")) for output_file in output_files
    ]

    # Don't even try to understand it
    for outputs, output_file in zip(output_versions, output_files):
        for output in outputs:
            image = images[output["name"]].copy()
            try:
                pictures[
                    f"{os.path.basename(output_file).replace('.json','')}-{output['name']}"
                ] = utils.show_dataset(image, output, ret=True, threshold=THRESHOLD)
            except Exception as exc:
                print(exc)


def create_app() -> FastAPI:
    app = FastAPI()
    load_v2()
    app.add_api_route("/", index)
    app.add_api_route("/image/{file_name}", serve_image_file)
    return app
