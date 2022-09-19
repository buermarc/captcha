import pytorch_lightning as pl
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.resnet import ResNet50_Weights


class CustomRcnnLightningModel(pl.LightningModule):

    def __init__(self, num_classes: int = 62+1, pretrained: bool = False):
        super().__init__()

        self.model = fasterrcnn_resnet50_fpn_v2(
            weights_backbone=ResNet50_Weights.DEFAULT,
            num_classes=num_classes
        )
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        self._pretrained = pretrained
        self._in_features = in_features
        self._num_classes = num_classes

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
        self.log("val_loss_mean", torch.mean(torch.Tensor(validation_step_outputs)))

    def configure_optimizers(self):
        SGD_kwargs = {"lr": 0.005, "momentum": 0.9, "weight_decay": 0.005}
        StepLR_kwargs = {"step_size": 3, "gamma": 0.7}
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
