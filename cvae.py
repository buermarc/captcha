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
from typing import List
import torch
import torch.nn as nn


def idx2onehot(idx, n):
    assert torch.max(idx).item() < n
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)

    onehot = torch.zeros(idx.size(0), n)
    onehot.scatter_(1, idx, 1)

    return onehot


class CVAE(nn.Module):

    def __init__(
        self,
        encoder_layer_sizes,
        decoder_layer_sizes, latent_dim,
        num_labels: int = 5,
        conditional=False,
        device: str = "gpu"
    ):
        """
        Arguments:
            encoder_layer_sizes (list[int]): list of the sizes of the encoder layers,
            decoder_layer_sizes (list[int]): list of the sizes of the decoder layers,
            latent_dim (int): dimension of latent space/bottleneck,
            num_labels (int): amount of labels (important for conditional VAE),,
            conditional (bool): True if CVAE, else False
        """

        super(CVAE, self).__init__()

        self.device = device
        self.latent_dim = latent_dim
        self.num_labels = num_labels
        self.encoder = Encoder(
            encoder_layer_sizes,
            latent_dim,
            num_labels,
            conditional,
            device
        )
        self.decoder = Decoder(
            decoder_layer_sizes,
            latent_dim,
            num_labels,
            conditional,
            device
        )

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
        batch_size = x.size(0)
        # x = x.view(-1,3*128*128)
        means, log_vars = self.encoder(x, c)
        sd = torch.exp(log_vars)
        epsilons = torch.randn((batch_size, self.latent_dim)).to(self.device)
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
        lat_vec = torch.randn((n, self.latent_dim)).to(self.device)
        x_sampled = self.decoder(lat_vec, c)
        return x_sampled


class Encoder(nn.Module):

    def __init__(
        self,
        layer_sizes: List[int],
        latent_dim: int,
        num_labels: int,
        conditional: bool = False,
        device: str = "gpu"
    ):
        super(Encoder, self).__init__()
        """
        Arguments:
            layer_sizes (list[int]): list of sizes of layers of the encoder,
            latent_dim (int): dimension of latent space, i.e. dimension out output of the encoder,
            num_labels (int): amount of labels,
            conditional (bool): True if CVAE and False if VAE
        """
        self.device = device
        self.conditional = conditional

        if self.conditional:
            layer_sizes[0] += num_labels


        '''
        input: WxH 30x60  # ?any norm?
        3x30x60 => conv-stride => 8x15x30 // ReLu
        8x15x30 => normal conv => 16x15x30 // ReLu
        16x15x30 => fc => 7200
        '''

        _ = [
            nn.Conv2d(3, 8, stride=2, padding=2, kernel_size=(3, 3)),
        ]

        self.network = nn.Sequential()

        # TODO rework
        image_size = 512
        input_size = image_size + num_labels

        self.network.add_module(name="N1", module=nn.Conv2d(3, 8, stride=2, padding=1, kernel_size=(3,3)))
        self.network.add_module(name="R1", module=nn.ReLU())
        self.network.add_module(name="N1a", module=nn.Conv2d(8, 8, padding=1, kernel_size=(3,3)))
        self.network.add_module(name="R2", module=nn.ReLU())

        self.network.add_module(name="N2", module=nn.Conv2d(8, 16, stride=2, padding=1, kernel_size=(3,3)))
        self.network.add_module(name="R3", module=nn.ReLU())
        self.network.add_module(name="N2a", module=nn.Conv2d(16, 16, padding=1, kernel_size=(3,3)))
        self.network.add_module(name="R3a", module=nn.ReLU())

        self.network.add_module(name="N3", module=nn.Conv2d(16, 32, stride=2, padding=1, kernel_size=(3,3)))
        self.network.add_module(name="R4", module=nn.ReLU())
        self.network.add_module(name="N3a", module=nn.Conv2d(32, 32, padding=1, kernel_size=(3,3)))
        self.network.add_module(name="R5", module=nn.ReLU())

        self.network.add_module(name="N4", module=nn.Conv2d(32, 64, stride=2, padding=1, kernel_size=(3,3)))
        self.network.add_module(name="R6", module=nn.ReLU())
        self.network.add_module(name="N4a", module=nn.Conv2d(64, 64, padding=1, kernel_size=(3,3)))
        self.network.add_module(name="R7", module=nn.ReLU())

        self.network.add_module(name="N5", module=nn.Conv2d(64, 64, stride=2, padding=1, kernel_size=(3,3)))
        self.network.add_module(name="R8", module=nn.ReLU())
        self.network.add_module(name="N5a", module=nn.Conv2d(64, 64, padding=1, kernel_size=(3,3)))
        self.network.add_module(name="R9", module=nn.ReLU())

        self.fcl = nn.Sequential(nn.Linear((16*16*64)+num_labels, 2048), nn.ReLU(), nn.Linear(2048, 2048), nn.ReLU())
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
            c = torch.zeros((batch_size), dtype=torch.int64)

        one_hot = idx2onehot(c, n=5).to(self.device)
        x = self.network(x)
        x = x.view(-1, 16*16*64)
        x_cat_y = torch.cat((x, one_hot), dim=1)
        x_y = self.fcl(x_cat_y)
        means = self.means_out(x_y)
        log_vars = self.log_sds_out(x_y)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(
        self,
        layer_sizes: List[int],
        latent_dim: int,
        num_labels: int,
        conditional: bool = False,
        device: str = "gpu"
    ):
        super(Decoder, self).__init__()
        """
        Arguments:
            layer_sizes (list[int]): list of sizes of layers of the decoder,
            latent_dim (int): dimension of latent space, i.e. dimension out
            input of the decoder,
            num_labels (int): amount of labels,
            conditional (bool): True if CVAE and False if VAE
        Output:
            x: Parameters of gaussian distribution; only mu (see above)
        """

        self.device = device
        self.conditional = conditional
        if self.conditional:
            latent_dim += num_labels

        self.network = nn.Sequential()
        layer_sizes.insert(0, latent_dim)

        last_ind = len(layer_sizes)-1


        self.fcl = nn.Sequential(nn.Linear(latent_dim, 2048), nn.ReLU(), nn.Linear(2048, 16*16*64), nn.ReLU())

        channel = 64

        self.network.add_module(name="T1", module=nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1))
        self.network.add_module(name="R1", module=nn.ReLU())

        self.network.add_module(name="T2", module=nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1))
        self.network.add_module(name="R2", module=nn.ReLU())

        self.network.add_module(name="T3", module=nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1))
        self.network.add_module(name="R3", module=nn.ReLU())

        self.network.add_module(name="T4", module=nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1))
        self.network.add_module(name="R4", module=nn.ReLU())

        self.network.add_module(name="T5", module=nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1, output_padding=1))
        self.network.add_module(name="R5", module=nn.ReLU())

        self.network.add_module(name="C6", module=nn.Conv2d(4, 3, kernel_size=(3, 3), padding=1))
        self.network.add_module(name="T6", module=nn.Tanh())

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
            c = torch.zeros((batch_size), dtype=torch.int64)

        one_hot = idx2onehot(c, n=5).to(self.device)
        z = torch.cat((z, one_hot), dim=1)

        z = self.fcl(z)
        z = z.view(-1, 64, 16, 16)
        x = self.network(z)
        return x


def loss_function(recon_x, x, mu, log_var):
    """
    Arguments:
        recon_x: reconstruced input
        x: input,
        mu, log_var: parameters of posterior (distribution of z given x)
    """
    x = x.view(-1, 3*512*512)
    recon_x = recon_x.view(-1, 3*512*512)

    msd = 1/2 * torch.sum((x-recon_x)**2, dim=1)
    kl_div = 1/2 * torch.sum(mu**2 + torch.exp(log_var)**2 - 2 * log_var - 1)
    kl_div_mean = kl_div / x.shape[0]
    msd_mean = torch.mean(msd)

    return (msd_mean + kl_div_mean)
