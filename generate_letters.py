from pythoncaptcha.image import ImageCaptcha
import shutil
import argparse
import random, string
import json
import os
import sys
from uuid import uuid4
import glob

fonts = glob.glob("pythoncaptcha/*.ttf")

image = ImageCaptcha()
BASEDIR = "data/letters/"


def generate_letter(
    labels: dict, outdir,
):
    # we only want one letter
    #letter = random.choice(string.ascii_letters + string.digits)
    letter = random.choice(string.digits)
    _uuid = str(uuid4())
    filename = f"{_uuid}-{letter}.png"

    image.write_letter(letter, f"{outdir}/{filename}")

    labels[filename.replace(".png", "")] = {
        "labels": letter,
    }

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
        for i in range(amount):
            labels = generate_letter(labels, outdir)

        if os.path.exists(label_file) and not args.delete:
            print("Label file already exists, --delete to override.")
            sys.exit(1)

        with open(label_file, mode="w+") as _file:
            json.dump(labels, _file, indent=2)
