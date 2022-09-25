import pytorch_lightning as pl
from torchvision.models.detection.faster_rcnn import FasterRCNN, fasterrcnn_mobilenet_v3_large_320_fpn, fasterrcnn_resnet50_fpn_v2, ResNet50_Weights, fasterrcnn_resnet50_fpn
import torch
import json
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights
import torchvision
from typing import List
from PIL import Image
import os


def evaluate_test_images(image_names: List[str], image_tensors: torch.Tensor, epoch: int, model, log_dir: str) -> None:
    outputs = model.forward(image_tensors)
    for name, _dict in zip(image_names, outputs):
        for key in _dict.keys():
            _dict[key] = _dict[key].tolist()
        _dict["name"] = name

    with open(log_dir + f"/epoch_{epoch}_test_output.json", mode="w+") as _file:
        json.dump(outputs, _file)


class CustomRcnnLightningModel(pl.LightningModule):

    def __init__(self, num_classes: int = 62+1, pretrained: bool = False):
        super().__init__()

        '''
        self.model_name = "fasterrcnn_mobilenet_v3_large_320_fpn"
        self.model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights_backbone=MobileNet_V3_Large_Weights.DEFAULT if pretrained else None,
            num_classes=num_classes,
            trainable_backbone_layers=3
        )
        '''
        self.model_name = "fasterrcnn_resnet50_fpn"
        self.model = fasterrcnn_resnet50_fpn(
            pretrained_backbone=pretrained,
            num_classes=num_classes,
            trainable_backbone_layers=3
        )
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        self._pretrained = pretrained
        self._in_features = in_features
        self._num_classes = num_classes

        image_files = os.listdir("./data/save_test/")
        image_tensors = torch.zeros((len(image_files), 3, 60, 160))
        for idx, image in enumerate(image_files):
            testimage = Image.open(f"./data/save_test/{image}")
            image_tensors[idx] = torchvision.transforms.ToTensor()(testimage)
        self.image_tensors = image_tensors.to("cuda").double()
        self.image_names = image_files

    def forward(self, image):
        self.model.eval()
        output = self.model(image)

        return output

    def training_step(self, batch, batch_idx):
        image, target = batch
        loss_dict = self.model(image, target)
        losses = sum(loss for loss in loss_dict.values())

        batch_size = len(batch[0])
        self.log_dict(loss_dict, batch_size=batch_size)
        self.log("train_loss", losses, batch_size=batch_size)
        return losses

    def validation_step(self, batch, batch_idx):
        image, target = batch
        output = self.model(image)

        metric = MeanAveragePrecision()
        metric.update(output, target)
        val_map = metric.compute()["map"]

        batch_size = len(batch[0])
        self.log("val_loss", val_map, batch_size=batch_size)

        return val_map

    def validation_epoch_end(self, validation_step_outputs):
        self.model.eval()
        evaluate_test_images(self.image_names, self.image_tensors, self.current_epoch, self.model, self.logger.log_dir)
        self.log("val_loss_mean", torch.mean(torch.Tensor(validation_step_outputs)))
        self.model.train()

    def configure_optimizers(self):
        SGD_kwargs = {"lr": 0.005, "momentum": 0.5, "weight_decay": 0.01}
        StepLR_kwargs = {"step_size": 2, "gamma": 0.75}
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, **SGD_kwargs)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **StepLR_kwargs)

        assert self.logger
        self.logger.log_hyperparams(SGD_kwargs)
        self.logger.log_hyperparams(StepLR_kwargs)
        self.logger.log_hyperparams({
            "in_features": self._in_features,
            "num_classes": self._num_classes,
            "pretrained": self._pretrained
        })

        return [optimizer], [lr_scheduler]
