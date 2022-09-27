#!/usr/bin/env bash
if [ -z "$1"  ]; then
    echo "args1: amount_train"
    exit 1
fi

if [ -z "$2"  ]; then
    echo "args1: amount_val"
    exit 1
fi

mkdir -p TextRecognitionDataGenerator/out
cp data/train_labels.json ./TextRecognitionDataGenerator/out/labels.json
pushd TextRecognitionDataGenerator
trdg -c $1 -rs -let -num -w 1 -e png -k 15 -rk -obb 1 -m 0 -d 1 -b 0 -wd 160 -cs  -al 1 -f 60
popd
mv TextRecognitionDataGenerator/out/*.png data/train
mv TextRecognitionDataGenerator/out/labels.json data/train_labels.json

mkdir -p TextRecognitionDataGenerator/out
cp data/val_labels.json ./TextRecognitionDataGenerator/out/labels.json
pushd TextRecognitionDataGenerator
trdg -c $2 -rs -let -num -w 1 -e png -k 15 -rk -obb 1 -m 0 -d 1 -b 0 -wd 160 -cs  -al 1 -f 60
popd
mv TextRecognitionDataGenerator/out/*.png data/val
mv TextRecognitionDataGenerator/out/labels.json data/val_labels.json
