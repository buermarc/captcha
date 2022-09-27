import glob
import os
import sys
import string

import torch
from torchvision.utils import save_image

import utils
from tqdm import tqdm
from cvae import CVAE

VERSION = 49
LATENT_DIM = 350

DATADIR = "data/letters/"
PREFERRED_DATATYPE = torch.double
BATCH_SIZE = 2
NUM_CLASSES = 62
DEVICE_STR = "cuda"

if __name__ == "__main__":
    model_path = glob.glob(f"cvae_tb_logs/CVAE/version_{VERSION}/models/*.ckpt")[0]
    model = CVAE.load_from_checkpoint(
        model_path, latent_dim=LATENT_DIM, num_classes=62, channel_width_height=(3, 30, 60)
    )
    model.device_str = DEVICE_STR
    model.encoder.device_str = DEVICE_STR
    model.decoder.device_str = DEVICE_STR
    model.to(DEVICE_STR).to(PREFERRED_DATATYPE)

    letters = string.ascii_letters + string.digits
    '''

    try:
        os.mkdir("sample_cvae_images")
    except Exception as exc:
        print(exc)
    labels = torch.Tensor(utils.encode_label([str(letter) for letter in [*letters]]))
    labels = labels.reshape(labels.shape[0], 1).to("cuda").to(torch.int64)

    result = model.sampling(n=62, c=labels)

    for res, letter in zip(result, letters):
        with open(f"sample_cvae_images/{letter}.png", mode="wb+") as _file:
            save_image(res, _file)
    '''

    '''
     checkpoint_file = "/home/ubuntu/data/2022-06-29T20:11+00:00-cvae_models/cvea-epoch-0048-data-00384.pt"
        checkpoint = torch.load(checkpoint_file)
        cvae.load_state_dict(checkpoint)

        class GenerativeClassifier(nn.Module):
            def __init__(self, model):
                super(GenerativeClassifier, self).__init__()
                self.model = model
            def forward(self, x):
                scores = []
                for inst in range(x.shape[0]):
                    score = []
                    for i in range(5):
                        label = i * torch.ones(1, dtype=torch.long)
                        out = cvae(x[inst].view(1, image_width_height*image_width_height), c=label)
                        loss = loss_function(out[0], x[inst].view(1, image_width_height*image_width_height), out[1], out[2]).item()
                        score.append(loss)
                    scores.append(score)
                scores = torch.tensor(scores)
                # choose class that minimizes loss
                classes = torch.argmin(scores, dim=1)
                scores = []
                return classes

        classifier = GenerativeClassifier(cvae)

        right = 0
        comp = 0
        for data in val_loader:
            cvae.eval()
            x, y = data
            x = x.view(-1, image_width_height*image_width_height)
            x = x.to(device)
            pred = classifier(x)
            # number of elements in batch
            comp += x.shape[0]
            # number of correctly classified instances
            right += torch.sum(pred==y)

        print(f'accuracy: {right/comp:.3f}')
    '''

    from pathlib import Path
    from captcha_dataset import CaptachDataset
    from torch.utils.data import DataLoader
    from typing import Tuple
    import os


    def adapted_loss_function(recon_x, x, mu, log_var, channel_width_height: Tuple[int, int, int]):
        """
        Arguments:
            recon_x: reconstruced input
            x: input,
            mu, log_var: parameters of posterior (distribution of z given x)
        """
        channel, width, height = channel_width_height  # 30, 60
        x = x.view(-1, channel*height*width)
        recon_x = recon_x.view(-1, channel*height*width)

        msd = 1/2 * torch.sum((x-recon_x)**2, dim=1)
        kl_div = 1/2 * torch.sum(mu**2 + torch.exp(log_var)**2 - 2 * log_var - 1, dim=1)
        # kl_div_mean = kl_div / x.shape[0]
        # msd_mean = torch.mean(msd)

        # return (msd_mean + kl_div_mean)
        return (msd + kl_div)  # return element wise loss

    core_count = os.cpu_count()
    letter_accuracy = []
    mat = torch.zeros((62, 62))
    for letter in tqdm(letters):
        test_dataset = CaptachDataset(
            image_path=Path(DATADIR + f"test-{str(letter)}/"),
            label_file=Path(DATADIR + "test_labels.json"),
            preferred_datatyp=PREFERRED_DATATYPE,
            is_captcha=False,
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=core_count if core_count else 4
        )

        prediction = []
        for batch in test_dataloader:
            images, labels = batch
            batch_size = images.shape[0]
            for idx in range(batch_size):
                # take image and 
                rng = torch.arange(1, NUM_CLASSES+1).reshape(NUM_CLASSES, 1).to(DEVICE_STR).to(torch.int64)
                image, label = images[idx], labels[idx]
                images = image.repeat(NUM_CLASSES, 1, 1, 1).to(DEVICE_STR).to(PREFERRED_DATATYPE)
                recon_batch, mu, log_var = model(images, rng)
                losses = adapted_loss_function(recon_batch, images, mu, log_var, (3, 60, 30))
                predicted_class = torch.argmin(losses).to("cpu")
                mat[utils.encode_label(letter)-1][predicted_class] += 1
                prediction.append(predicted_class == label-1)  # label from utils is +1

        letter_accuracy.append((sum(prediction)/len(prediction)).tolist()[0])

    import json
    out_json = {}

    for letter, accuracy in zip(*[letters], letter_accuracy):
        out_json[str(letter)] = accuracy

    with open("letter_accuracy.json", mode="w+") as _file:
        json.dump(out_json, _file)

    torch.save(mat, "heatmap.pt")
