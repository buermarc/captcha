from pythoncaptcha.image import ImageCaptcha
import shutil
import argparse
import random, string
import json
import os
import sys
from cvae import CVAE
import glob

from sample_cvae import DEVICE_STR, LATENT_DIM, PREFERRED_DATATYPE

image = ImageCaptcha()
BASEDIR = "data-generated-with-cvae/"
VERSION = 49


def load_model(version: int) -> CVAE:
    checkpoint_file = glob.glob(f"cvae_tb_logs/CVAE/version_{version}/models/*.ckpt")[0]
    model = CVAE.load_from_checkpoint(
        checkpoint_file,
        latent_dim=LATENT_DIM,
        num_classes=62,
        channel_width_height=(3, 30, 60),
    )
    model.device_str = DEVICE_STR
    model.to(PREFERRED_DATATYPE)
    model.to(DEVICE_STR)
    return model


def generate_python_captcha(
    labels: dict, outdir, model, number_of_chars: int = 0
):
    if number_of_chars == 0:
        number_of_chars = random.randrange(3) + 4
    content = "".join(
        random.choice(string.ascii_letters + string.digits)
        for _ in range(number_of_chars)
    )  # take uppercase only to reduce similar chars like w and W

    _ = image.write_with_cvae(content, f"{outdir}/{content}.png", model)

    '''
    labels[content] = {
        "boxes": [[x1, y1, x2, y2] for (x1, y1), (x2, y2) in offset_points],
        "labels": [*content],
    }
    '''

    return labels


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate catpchas")
    parser.add_argument(
        "amount",
        metavar="N",
        type=int,
        nargs="?",
        help="Amount of to be generated captchas",
        default=6,
    )
    parser.add_argument(
        "--val",
        action="store_true",
        help="Generate validation data",
        default=False,
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Generate test data",
        default=False,
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Generate train data",
        default=False,
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete existing data",
        default=False,
    )
    args = parser.parse_args()

    if args.val and args.test:
        print("Either val or test not both!")
        sys.exit(1)

    datasets = []

    if args.train:
        datasets.append(
            (BASEDIR + "train", BASEDIR + "train_labels.json", args.amount)
        )
    elif args.val:
        datasets.append(
            (BASEDIR + "val", BASEDIR + "val_labels.json", args.amount)
        )
    elif args.test:
        datasets.append(
            (BASEDIR + "test", BASEDIR + "test_labels.json", args.amount)
        )

    else:
        # Generate everything
        train_amount = int(args.amount)
        val_amount = int(args.amount * 0.2)
        test_amount = int(args.amount * 0.2)
        datasets.extend(
            [(BASEDIR + "train", BASEDIR + "train_labels.json", train_amount),
             (BASEDIR + "val", BASEDIR + "val_labels.json", val_amount),
             (BASEDIR + "test", BASEDIR + "test_labels.json", test_amount)]
        )

    for dataset in datasets:
        outdir, label_file, amount = dataset

        if os.path.exists(outdir) and not args.delete:
            print("Data already exists, --delete to override with new data.")
            sys.exit(1)

        if os.path.exists(outdir) and args.delete:
            shutil.rmtree(outdir)

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        labels = {}
        model = load_model(VERSION)
        for i in range(amount):
            labels = generate_python_captcha(labels, outdir, model)

        if os.path.exists(label_file) and not args.delete:
            print("Label file already exists, --delete to override.")
            sys.exit(1)

        with open(label_file, mode="w+") as _file:
            json.dump(labels, _file, indent=2)
