from pythoncaptcha.image import ImageCaptcha
import argparse
import random, string
import json
import os

image = ImageCaptcha()
BASEDIR = "data/"


def generate_python_captcha(
    labels: dict, number_of_chars: int = 0, boxes: bool = False
):
    if number_of_chars == 0:
        number_of_chars = random.randrange(3) + 4
    content = "".join(
        random.choice(string.ascii_letters + string.digits)
        for _ in range(number_of_chars)
    )  # take uppercase only to reduce similar chars like w and W

    offset_points = image.write(content, f"{OUTDIR}/{content}.png", boxes=boxes)

    labels[content] = {
        "boxes": [[x1, y1, x2, y2] for (x1, y1), (x2, y2) in offset_points],
        "labels": [*content],
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
    args = parser.parse_args()

    OUTDIR = BASEDIR + "val" if args.val else BASEDIR + "train"

    if not os.path.exists(OUTDIR):
        os.mkdir(OUTDIR)

    labels = {}
    for i in range(args.amount):
        labels = generate_python_captcha(labels, boxes=False)

    with open(BASEDIR + f"{'val_' if args.val else 'train_'}labels.json", mode="w+") as _file:
        json.dump(labels, _file, indent=2)
