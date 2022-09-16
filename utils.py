from typing import List
from PIL import ImageFont
from PIL.ImageDraw import Draw

def print_dataset(dataset):
    boxes = dataset.get('boxes').detach().numpy()
    labels = dataset.get('labels').detach().numpy()
    scores = dataset.get('scores').detach().numpy()
    for label, box, score in zip(labels, boxes, scores):
        print(f"Label: {label}, Box: {box}, Score: {score}\n")


def show_dataset(image, target):
    boxes = target.get('boxes').detach().numpy()
    labels = target.get('labels').detach().numpy()
    scores = target.get('scores').detach().numpy()
    draw = Draw(image)
    for score, label, box in zip(scores, labels, boxes):
        draw.rectangle(box, outline="red")
        text = f"{label} {(score*100):.1f}%"
        font = ImageFont.truetype("pythoncaptcha/DroidSansMono.ttf", 30)
        draw.text(box[0:2], text, font=font, fill='red')
    image.show()

def decode_label(label: int):
    if isinstance(label, list):
        return [_decode_label(element) for element in label]
    return _decode_label(label)

def _decode_label(label: int):
    if 0 <= label < 36:
        if label < 10:
            return label
        return chr(label+55)
    raise "Label has to be between 0 and 35 (each incl)"

def encode_label(label: str):
    if isinstance(label, list):
        return [_encode_label(element) for element in label]
    return _encode_label(label)

def _encode_label(label: str):
    if 48 <= ord(label) <= 57 or 65 <= ord(label) <= 90:
        if ord(label) < 60:
            return ord(label)-48
        return ord(label)-55
    raise "Label has to be alphanumeric"