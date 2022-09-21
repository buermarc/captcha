'''
Different approaches to train the cvae

We need some form of training input:
    - whole captcha = image and string:
        - Allows to train on different captcha layouts as we do not need any
          bounding boxes compared to the "solving captchas" task
        - we could also train on the mnist dataset and use the generated
          letters to generate captachs similar to our algorithmic approach
            - generated images are hybrid machine learning and algorithmic:
        - instead of the mnist dataset we might use the different fonts and to
          generate simple letters and encode trying to achieve encoding the
          font type into the latent space allowing us to create letters that
          resemble a combination or merging of different fonts, which might
          make it harder for an classification network to correctly classify a
          letter
            - we should still only encode the letter not the font, the letter
              is a hard requirement that has to be met while the font can be
              something where a hard assignment to any fontset isn't required



Different CVAEs:
1. CVAE generating a whole captcha:
    - generating a whole captcha is a bit annoying because we have such a high
      class number:
classes = 26*2 (letters) + 10 (numbers) + 1 (space/emptiness)

we want to make captchas with e.g. max 6 numbers
63^6 -> 62523502209 (problematic)
Problem:
! For now let's just consider the first approach a bit more feasible

Problem is we need something that can produce the thing based on a a label
because the human has to assign the generated image to a class.

2. CVAE genearting only one letter at the time:
    - we need single letter but best already distorted
        - > adapt the data generation script

Broader Prolbems that should be considered:
    - What happens if the captcha is fillped, can we just rotate it mulitple
      times (steps of 20 deg) and then use the argmax(sum(score)) For this to
      work we would have to train with slightly augmented data (also rotate
      training data a little bit to make the deg window that we have to hit
      more likely)

'''
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pathlib import Path
import os
import utils
from torch.utils.data import DataLoader
from captcha_dataset import CaptachDataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchinfo import summary
from pytorch_lightning.callbacks import LearningRateMonitor
from torchvision.utils import save_image


PREFERRED_DATATYPE = torch.double
BATCH_SIZE = 2
DATADIR = "data/letters/"


def idx2onehot(idx, n, device_str):
    idx = idx - 1

    assert torch.max(idx).item() < n
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)

    onehot = torch.zeros(idx.size(0), n).to(device_str)
    onehot.scatter_(1, idx, 1)

    return onehot


class CVAE(pl.LightningModule):

    def __init__(
        self,
        latent_dim,
        num_classes: int = 5,
        device: str = "cuda"
    ):
        """
        Arguments:
            latent_dim (int): dimension of latent space/bottleneck,
            num_classes (int): amount of labels,
        """

        super(CVAE, self).__init__()

        self.model_name = "CVAE-V1"
        self.device_str = device
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.encoder = Encoder(
            latent_dim,
            num_classes,
            device
        )
        self.decoder = Decoder(
            latent_dim,
            num_classes,
            device
        )
        self.static_labels = ["Z", "0", "v", "w", "n", "q", "7", "8", "I", "l"]
        self.static_encoded_labels = utils.encode_label(
            ["Z", "0", "v", "w", "n", "q", "7", "8", "I", "l"]
        )
        self.static_latent = torch.randn(
            (3, self.latent_dim)
        ).to(self.device_str).to(torch.double)

    def forward(self, x, c=None):
        """
        Forward Process of whole VAE/CVAE.
        Arguments:
            x: tensor of dimension (batch_size, 1, 28, 28) or (batch_size, 28*28)
            c: None or tensor of dimension (batch_size, 1)
        Output: recon_x, means, log_var
            recon_x: see explanation on second part of estimator above,
            means: output of encoder,
            log_var: output of encoder (logarithm of variance)
        """
        batch_size = x.shape[0]
        means, log_vars = self.encoder(x, c)
        sd = torch.exp(log_vars)
        epsilons = torch.randn((batch_size, self.latent_dim)).to(self.device_str)
        z = sd * epsilons + means
        recon_x = self.decoder(z, c)

        return recon_x, means, log_vars

    def sampling(self, n=2, c=None):
        """
        Generates new samples by feeding a random latent vector to the decoder.
        Arguments:
            n (int): amount of samples
            c      : None or tensor of dimension (batch_size, 1) (labels to condition on)
        Output:
            x_sampled: n randomly sampled elements of the output distribution
        """
        lat_vec = torch.randn((n, self.latent_dim)).to(self.device_str)
        x_sampled = self.decoder(lat_vec, c)
        return x_sampled

    def training_step(self, batch, batch_idx):
        image, target = batch
        batch_size = image.shape[0]

        recon_batch, mu, log_var = self(image, target)
        loss = loss_function(recon_batch, image, mu, log_var)
        train_loss = loss.item()
        self.log("train_loss", train_loss, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        image, target = batch
        batch_size = image.shape[0]

        recon_batch, mu, log_var = self(image, target)
        loss = loss_function(recon_batch, image, mu, log_var)
        val_loss = loss.item()
        self.log("val_loss", val_loss, batch_size=batch_size)
        return loss

    def validation_epoch_end(self, _validation_step_outputs):
        image_dir = f"{self.logger.log_dir}/sampled_images"

        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        for label, elabel in zip(self.static_labels, self.static_encoded_labels):
            images = self.decoder(self.static_latent, torch.ones(len(self.static_latent)).to(self.device_str).to(torch.int64) * elabel)

            for idx, image in enumerate(images):
                with open(
                    f"{image_dir}/epoch-{self.current_epoch}-lat-{idx}-label-{label}.png",
                    mode="wb+"
                ) as _file:
                    save_image(image, _file, format="png")

    def configure_optimizers(self):
        Adam_kwargs = {"lr": 0.005, "weight_decay": 0.01}
        StepLR_kwargs = {"step_size": 2, "gamma": 0.7}

        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, **Adam_kwargs)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **StepLR_kwargs)

        assert self.logger
        self.logger.log_hyperparams(Adam_kwargs)
        self.logger.log_hyperparams(StepLR_kwargs)
        self.logger.log_hyperparams({
            "latent_dim": self.latent_dim,
            "num_classes": self.num_classes,
        })

        return [optimizer], [lr_scheduler]


class Encoder(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        device: str = "cuda"
    ):
        super(Encoder, self).__init__()
        """
        Arguments:
            latent_dim (int): dim of latent space,
            num_classes (int): amount of labels,
        """
        self.device_str = device
        self.num_classes = num_classes

        args = [
            nn.Conv2d(4, 8, stride=2, padding=1, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(8, 16, stride=1, padding=1, kernel_size=(3, 3)),
            nn.ReLU(),
        ]

        self.network = nn.Sequential(*args)

        self.fcl = nn.Sequential(
            nn.Linear((16*15*30)+num_classes, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
        )

        self.means_out = nn.Linear(2048, latent_dim)
        self.log_sds_out = nn.Linear(2048, latent_dim)

    def forward(self, x, c=None):
        """
        Arguments:
            x: tensor of dimension (C, W, H) or C*W*H
            c: None or tensor of dimension (batch_size, 1) ???
        Output:
            means: tensor of dimension (batch_size, latent_dim),
            log_var: tensor of dimension (batch_size, latent_dim)
        """
        # TODO check if batch_size is correct
        batch_size = x.shape[0]

        if c is None:
            c = torch.ones((batch_size), dtype=torch.int64)

        one_hot = idx2onehot(c, n=self.num_classes, device_str=self.device_str).to(self.device_str)
        x = self.network(x)
        x = x.view(-1, 16*15*30)
        x_cat_y = torch.cat((x, one_hot), dim=1)
        x_y = self.fcl(x_cat_y)
        means = self.means_out(x_y)
        log_vars = self.log_sds_out(x_y)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        device: str = "cuda"
    ):
        super(Decoder, self).__init__()
        """
        Arguments:
            latent_dim (int): dimension of latent space, i.e. dimension out
            input of the decoder,
            num_classes (int): amount of labels,
        Output:
            x: Parameters of gaussian distribution; only mu (see above)
        """

        self.device_str = device
        self.num_classes = num_classes

        latent_dim += num_classes

        self.fcl = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 16*15*30),
            nn.ReLU(),
        )

        args = [
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 4, stride=1, padding=1, kernel_size=(3, 3)),
            nn.Tanh(),
        ]

        self.network = nn.Sequential(*args)

    def forward(self, z, c=None):
        """
        Argumetns:
            z: tensor of dimension (batch_size, latent_dim)
            c: None or tensor of dimension (batch_size, 1)
        Outputs:
            x: mu of gaussian distribution (reconstructed image from latent code z)
        """
        batch_size = z.shape[0]

        if c is None:
            c = torch.ones((batch_size), dtype=torch.int64)

        one_hot = idx2onehot(c, n=self.num_classes, device_str=self.device_str).to(self.device_str)
        z = torch.cat((z, one_hot), dim=1)

        z = self.fcl(z)
        z = z.view(-1, 16, 30, 15)
        x = self.network(z)
        return x


def loss_function(recon_x, x, mu, log_var):
    """
    Arguments:
        recon_x: reconstruced input
        x: input,
        mu, log_var: parameters of posterior (distribution of z given x)
    """
    x = x.view(-1, 4*60*30)
    recon_x = recon_x.view(-1, 4*60*30)

    msd = 1/2 * torch.sum((x-recon_x)**2, dim=1)
    kl_div = 1/2 * torch.sum(mu**2 + torch.exp(log_var)**2 - 2 * log_var - 1)
    kl_div_mean = kl_div / x.shape[0]
    msd_mean = torch.mean(msd)

    return (msd_mean + kl_div_mean)


if __name__ == '__main__':
    core_count = os.cpu_count()

    train_data = CaptachDataset(
        image_path=Path(DATADIR + "train"),
        label_file=Path(DATADIR + "train_labels.json"),
        preferred_datatyp=PREFERRED_DATATYPE,
        is_captcha=False,
    )
    val_data = CaptachDataset(
        image_path=Path(DATADIR + "val"),
        label_file=Path(DATADIR + "val_labels.json"),
        preferred_datatyp=PREFERRED_DATATYPE,
        is_captcha=False,
    )

    train_dataloader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=core_count if core_count else 4
    )

    val_dataloader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=core_count if core_count else 4
    )

    model = CVAE(latent_dim=2048, num_classes=62+1)
    # summary(model, device='cuda', input_size=(BATCH_SIZE, 4, 60, 30))

    logger = TensorBoardLogger("cvae_tb_logs", name="CVAE")
    logger.log_hyperparams({
        "batch_size": BATCH_SIZE,
        "train_data_size": len(train_data),
        "val_data_size": len(val_data),
        "model_name(selfset)": model.model_name
    })

    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=2,
        patience=3,
        mode="min",
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{logger.log_dir}/models/",
        filename="{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        verbose=True
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[early_stopping, lr_monitor, checkpoint_callback],
        max_epochs=100,
        logger=logger,
        log_every_n_steps=50,
    )

    trainer.fit(model.to(PREFERRED_DATATYPE), train_dataloader, val_dataloader)
