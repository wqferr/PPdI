#!/usr/bin/env bash
python generatevgg16.py
python generatedata.py ../Data/Datasets/separated/train/ 64 64 ../Data/Datasets/keras/train
python generatedata.py ../Data/Datasets/separated/test/ 64 64 ../Data/Datasets/keras/test 0 1