import pytorch_lightning as pl
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
import torch
import torchmetrics


class CustomRcnnLightningModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)

        self.val_map = torchmetrics.Accuracy()

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

        val_map = self.val_map(output, target)
        self.log("val_map", val_map["map"])

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        return [optimizer], [lr_scheduler]
