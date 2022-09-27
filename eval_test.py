'''
load test data
for item in test data process through model
add to solution json
    solution json holds information needed for own metric
also need map metric and also need and levensthein

input map ->
input own ->
input lev ->
'''

from PIL import Image
from tqdm import tqdm
import json
import os
from os.path import basename
import torch
from custom_rcnn_lightning_model import CustomRcnnLightningModel
from custom_rcnn_lightning_model_v2 import CustomRcnnLightningModelV2

from torchvision.io import read_image
from Levenshtein import distance as levenshtein_distance
import numpy as np
import glob
import matplotlib.pyplot as plt

PREFERRED_DATATYPE = torch.double
BATCH_SIZE = 2
DATADIR = "data/"
OUTDIR = "out_eval"
if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)

SAMPLES = np.random.choice(np.arange(50), 3, replace=False)


from utils import _sort_labels_threshold, _check_letters_correct_number, _check_letters_correct_number, _check_letters_wrong_number, show_dataset


def own_testmetric_2(correct_labels, datasets, threshold: float = 0.0):
    if len(correct_labels) is not len(datasets):
        raise ValueError("Number of correct labels and datasets are not equal")

    correct_detections = 0
    correct_letters_list = []
    lv_distances = []
    for correct_label, dataset in zip(correct_labels, datasets):
        if len(dataset.get('boxes')) > 0:
            labels_sorted = _sort_labels_threshold(dataset, threshold)
            lv_d = levenshtein_distance(correct_label, labels_sorted)
            lv_distances.append(lv_d)
            correct_detection = False
            if len(correct_label) == len(labels_sorted):
                correct_detection = True if lv_d == 0 else False
                correct_letters = lv_d / len(correct_label)
            else:
                if len(correct_labels) > len(labels_sorted):
                    correct_letters = (len(correct_labels)-lv_d) / len(correct_labels)
                else:
                    correct_letters = (len(labels_sorted)-lv_d) / len(labels_sorted)
            correct_letters_list.append(correct_letters)
            correct_detections += int(correct_detection)
        else:
            correct_letters_list.append(0)
            lv_distances.append(7)

    letters_mean = np.mean(correct_letters_list)
    letters_std = np.std(correct_letters_list)
    lv_mean = np.mean(lv_distances)
    correct_percentage = correct_detections / len(correct_labels)
    return correct_percentage, letters_mean, letters_std, lv_mean

def own_testmetric(correct_labels, datasets, threshold: float = 0.0):
    if len(correct_labels) is not len(datasets):
        raise ValueError("Number of correct labels and datasets are not equal")

    correct_detections = 0
    correct_letters_list = []
    lv_distances = []
    for correct_label, dataset in zip(correct_labels, datasets):
        if len(dataset.get('boxes')) > 0:
            labels_sorted = _sort_labels_threshold(dataset, threshold)
            lv_distances.append(levenshtein_distance(correct_label, labels_sorted))
            correct_detection = False
            if len(correct_label) == len(labels_sorted):
                correct_letters, correct_detection = _check_letters_correct_number(correct_label, labels_sorted)
            else:
                correct_letters = _check_letters_wrong_number(correct_label, labels_sorted)
            correct_letters_list.append(correct_letters)
            correct_detections += int(correct_detection)
        else:
            correct_letters_list.append(0)
            lv_distances.append(7)

    letters_mean = np.mean(correct_letters_list)
    letters_std = np.std(correct_letters_list)
    lv_mean = np.mean(lv_distances)
    correct_percentage = correct_detections / len(correct_labels)
    return correct_percentage, letters_mean, letters_std, lv_mean


DEVICE_STR = "cuda"

def eval_model_against_dataset(checkpoint_file: str, datadir: str, dataset_name: str, model_name: str, model_class) -> str:

    model = model_class.load_from_checkpoint(checkpoint_file).to(PREFERRED_DATATYPE)

    model.device_str = DEVICE_STR
    model.eval()

    outputs = {}
    json_outputs = {}

    files = glob.glob(f"{datadir}/*.png")
    if len(files) == 0:
        breakpoint()

    for file in tqdm(files):
        image = read_image(file).to(PREFERRED_DATATYPE) / 255
        content = basename(file).replace(".png", "")

        image = image.reshape(1, *tuple(image.shape))

        output = model(image)[0]

        data = {
            "boxes": output["boxes"].detach().numpy(),
            "labels": output["labels"].detach().numpy(),
            "scores": output["scores"].detach().numpy(),
        }
        json_data = {
            "boxes": data["boxes"].tolist(),
            "labels": data["labels"].tolist(),
            "scores": data["scores"].tolist(),
        }

        del image
        del output
        outputs[content] = data
        json_outputs[content] = json_data

    json_path = f"{OUTDIR}/{dataset_name}-{model_name}---{DATADIR.replace('/', '_')}-{checkpoint_file.replace('/', '-')}.json".replace(" ", "")
    with open(json_path, mode="w+") as _file:
        json.dump(json_outputs, _file)

    keys = outputs.keys()
    values = outputs.values()

    if len(keys) != len(values):
        breakpoint()

    correct_percentages = []
    letters_means = []
    lv_means = []

    x_ = np.arange(0, 1.1, 0.01)
    for i in x_:
        correct_percentage, letters_mean, _, lv_mean = own_testmetric_2(
            keys,
            values,
            i
        )
        correct_percentages.append(correct_percentage)
        letters_means.append(letters_mean)
        lv_means.append(lv_mean)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Total and Letter Correct", color="black")

    a, = ax1.plot(x_, correct_percentages, label="Total Correct", color="tab:green")
    b, = ax1.plot(x_, letters_means, label="Letter Correct", color="tab:blue")
    ax1.tick_params(axis = "y", labelcolor="black")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Levenshtein", color="tab:orange")
    c,  = ax2.plot(x_, lv_means, label="Levenshtein", color="tab:orange")
    ax2.tick_params(axis = "y", labelcolor="tab:orange")
    ax2.set_ylim([0, max(lv_means)+5])

    p = [a, b, c]
    ax1.legend(p, [p_.get_label() for p_ in p], loc= 'upper center', fontsize= 'small')

    fig.suptitle(f"{dataset_name} - {model_name}")

    plotfilename = f"{OUTDIR}/{dataset_name}-{model_name}---{DATADIR.replace('/', '_')}-{checkpoint_file.replace('/', '-')}.pdf".replace(" ", "")
    plt.savefig(plotfilename)

    threshold = max(letters_means)

    # samples = np.random.choice(files, 3, replace=False)

    for idx in SAMPLES:
        file = files[idx]
        image = Image.open(file)
        content = basename(file).replace(".png", "")
        output = outputs[content]
        output["boxes"] = output["boxes"].tolist()
        output["labels"] = output["labels"].tolist()
        output["scores"] = output["scores"].tolist()
        im = show_dataset(image, output, threshold=threshold, ret=True)
        im.save(f"{OUTDIR}/{dataset_name}-{model_name}-{idx}.png".replace(" ", ""))

    return plotfilename


if __name__ == '__main__':
    # mutli font, normal font, nicolas_1, nicolas_2
    # CustomRcnnLightningModel_v1_finetuned, CustomRcnnLightningModel_v1_not_finetuned, CustomRcnnLightningModel_v2
    for datadir, dataset_name in [
        ("data/subset-test-multifont", "Python Multifont"),
        ("data/subset-test-singlefont", "Python Singlefont"),
        ("../data-captcha/subset-gregwar", "Gregwar"),
        ("../data-captcha/subset-wilhelmy", "Wilhelmy")
    ]:
        for checkpoint_file, model_name, model_class in [
            (
                glob.glob(f"tb_logs/CustomRcnnLightningModel-Continue/version_2/models/*.ckpt")[0],
                "Faster R-CNN Finetuned",
                CustomRcnnLightningModel,
            ),
            (
                glob.glob(f"tb_logs/CustomRcnnLightningModel/version_2/models/*.ckpt")[0],
                "Faster R-CNN",
                CustomRcnnLightningModel,
            ),
            (
                glob.glob("../captcha/tb_logs/CustomRcnnLightningModel/version_59/models/epoch=09-val_loss=0.63.ckpt")[0],
                "Faster R-CNN V2",
                CustomRcnnLightningModelV2,
            )
        ]:
            print(eval_model_against_dataset(checkpoint_file, datadir, dataset_name, model_name, model_class))
