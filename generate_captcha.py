from pythoncaptcha.image import ImageCaptcha
import argparse
import random, string
import json

image = ImageCaptcha()

def generate_python_captcha(labels: dict, number_of_chars: int = 0, boxes: bool = False):
    if(number_of_chars == 0):
        number_of_chars = random.randrange(2)+4
    content = "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(number_of_chars)) # take uppercase only to reduce similar chars like w and W
    #data = image.generate(content)

    offset_points = image.write(
        content,
        f"captchas/{content}.png", boxes=boxes
    )

    labels[content] = [
        {"class": content[idx], "x1": x1, "y1": y1, "x2": x2, "y2": y2}
        for idx, ((x1, y1), (x2, y2)) in enumerate(offset_points)]

    return labels


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate catpchas')
    parser.add_argument('amount', metavar='N', type=int, nargs='?', help='Amount of to be generated captchas', default=5)
    args = parser.parse_args()

    labels = {}
    for i in range(args.amount):
        labels = generate_python_captcha(labels, boxes=True)

    with open("labels.json", mode="w+") as _file:
        json.dump(labels, _file, indent=4)
