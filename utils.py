import json, torch
from typing import Optional
import numpy as np
from PIL import ImageFont
from PIL.ImageDraw import Draw
from PIL.Image import Image


def print_dataset(dataset):
    boxes = dataset.get('boxes').detach().numpy()
    labels = dataset.get('labels').detach().numpy()
    scores = dataset.get('scores').detach().numpy()
    for label, box, score in zip(labels, boxes, scores):
        print(f"Label: {decode_label(label)}, Box: {box}, Score: {score}\n")


def show_dataset(image, target, threshold: float = 0.0, ret: bool = False) -> Optional[Image]:
    boxes = target.get('boxes').detach().numpy()
    labels = target.get('labels').detach().numpy()
    scores = target.get('scores').detach().numpy()
    draw = Draw(image)
    for score, label, box in zip(scores, labels, boxes):
        if score > threshold:
            draw.rectangle(box, outline="red")
            text = f"{decode_label(label)} {(score*100):.0f}%"
            font = ImageFont.truetype("pythoncaptcha/DroidSansMono-V4.ttf", 10)
            draw.text(box[0:2], text, font=font, fill='red')
    if ret:
        return image
    image.show()

def decode_label(label):
    if isinstance(label, list) or isinstance(label, np.ndarray):
        return [_decode_label(element) for element in label]
    return _decode_label(label)

def _decode_label(label):
    if not isinstance(label, int):
        label = int(label)
    if 0 <= label < 62:
        if label < 10:
            return label
        if label < 36:
            return chr(label+55)
        return chr(label+61)
    raise ValueError("Label has to be between 0 and 61 (each incl)")

def encode_label(label: str):
    if isinstance(label, list):
        return [_encode_label(element) for element in label]
    return _encode_label(label)

def _encode_label(label: str):
    if 48 <= ord(label) <= 57 or 65 <= ord(label) <= 90 or 97 <= ord(label) <= 122:
        if ord(label) < 60:
            return ord(label)-48
        if ord(label) < 95:
            return ord(label)-55
        return ord(label)-61
    raise ValueError("Label has to be alphanumeric")


def own_testmetric(correct_labels, datasets, threshold: float = 0.0):
    if len(correct_labels) is not len(datasets):
        raise ValueError("Number of correct labels and datasets are not equal")

    correct_detections = 0
    correct_letters_list = []
    for correct_label, dataset in zip(correct_labels, datasets):
        if len(dataset.get('boxes').detach().numpy()) > 0:
            labels_sorted = _sort_labels_threshold(dataset, threshold)
            correct_detection = False
            if len(correct_label) == len(labels_sorted):
                correct_letters, correct_detection = _check_letters_correct_number(correct_label, labels_sorted)
            else:
                correct_letters = _check_letters_wrong_number(correct_label, labels_sorted)
            correct_letters_list.append(correct_letters)
            correct_detections += int(correct_detection)
        else:
            correct_letters_list.append(0)
    
    letters_mean = np.mean(correct_letters_list)
    letters_std = np.std(correct_letters_list)
    correct_percentage = correct_detections / len(correct_labels)
    return correct_percentage, letters_mean, letters_std


def _check_letters_wrong_number(correct_label, labels_sorted):
    if len(labels_sorted) > len(correct_label):
        x = correct_label
        y = labels_sorted
    else:
        x = labels_sorted
        y = correct_label
    
    correct_letters = 0
    
    idy = 0
    for idx in range(len(x)):
        delta = 0
        correct, delta = _recursive_letter_check(x, y, idx, idy, delta)
        correct_letters += correct
        idy += 1+delta
    return correct_letters/len(y)


def _recursive_letter_check(x, y, idx, idy, delta):
    if idy+delta == len(y):
        return 0, delta-1
    else:
        if x[idx] == y[idy+delta]:
            return 1, delta
        else:
            return _recursive_letter_check(x, y, idx, idy, delta+1)


def _check_letters_correct_number(x, y):
    correct_letters = 0
    for letter_x, letter_y in zip(x,y):
        if letter_x == letter_y:
            correct_letters +=1
    correct_detection = True if correct_letters == len(x) else False
    return correct_letters/len(x), correct_detection


def _sort_labels_threshold(dataset, threshold):
    boxes = dataset.get('boxes').detach().numpy()
    labels = dataset.get('labels').detach().numpy()
    scores = dataset.get('scores').detach().numpy()

    boxes_upper_left = boxes[:, 0]
    index_order = np.argsort(boxes_upper_left)
    labels_sorted = labels[index_order]
    scores_sorted = scores[index_order]

    labels_concat = ""
    for score, label in zip(scores_sorted, decode_label(labels_sorted)):
        if score > threshold:
            labels_concat += str(label)
    return labels_concat


if __name__ == '__main__':
    with open("data/train_labels.json", mode="r") as _file:
        labels_json = json.load(_file)
    
    mock = [
    {
        'boxes': torch.Tensor(
            [[2.2495e+01, 2.8099e+00, 4.1562e+01, 5.6435e+01],
            [1.1614e+02, 2.4039e+00, 1.3830e+02, 5.6009e+01],
            [5.0569e+01, 8.1115e-02, 7.4713e+01, 5.9538e+01],
            [8.1298e+01, 1.9613e+00, 1.0305e+02, 5.6535e+01],
            [9.8708e+01, 0.0000e+00, 1.2088e+02, 5.9347e+01]]),
        'labels': torch.Tensor([23, 15,  9, 19, 21]),
        'scores': torch.Tensor([0.9952, 0.9949, 0.9898, 0.9891, 0.9889])
    },
    {
        'boxes': torch.Tensor(
            [[2.2495e+01, 2.8099e+00, 4.1562e+01, 5.6435e+01],
            [1.1614e+02, 2.4039e+00, 1.3830e+02, 5.6009e+01],
            [5.0569e+01, 8.1115e-02, 7.4713e+01, 5.9538e+01],
            [2.4039e+01, 8.1115e-02, 7.4713e+01, 5.9538e+01],
            [8.1298e+01, 1.9613e+00, 1.0305e+02, 5.6535e+01],
            [9.8708e+01, 0.0000e+00, 1.2088e+02, 5.9347e+01]]),
        'labels': torch.Tensor([23, 15,  9, 7, 19, 21]),
        'scores': torch.Tensor([0.9952, 0.9949, 0.9898, 0.4, 0.9891, 0.9889])
    },
    {
       'boxes': torch.Tensor(
            [[2.2495e+01, 2.8099e+00, 4.1562e+01, 5.6435e+01],
            [1.1614e+02, 2.4039e+00, 1.3830e+02, 5.6009e+01],
            [8.1298e+01, 1.9613e+00, 1.0305e+02, 5.6535e+01],
            [9.8708e+01, 0.0000e+00, 1.2088e+02, 5.9347e+01]]),
        'labels': torch.Tensor([23, 15, 19, 21]),
        'scores': torch.Tensor([0.9952, 0.9949, 0.9891, 0.9889])
    },
    {
       'boxes': torch.Tensor(
            [[9.8708e+01, 0.0000e+00, 1.2088e+02, 5.9347e+01]]),
        'labels': torch.Tensor([21]),
        'scores': torch.Tensor([0.9889])
    },
    {
       'boxes': torch.Tensor([]),
        'labels': torch.Tensor([]),
        'scores': torch.Tensor([])
    },
    ]

    print(own_testmetric(["N9JLF", "N9JLF", "N9JLF", "N9JLF", "N9JLF"], mock))
