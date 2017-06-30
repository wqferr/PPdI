#!/usr/bin/env bash
python generatevgg16.py 64 64 10
python generatedata.py ../Data/Datasets/filtered/train/ 64 64 ../Data/Datasets/keras/train
python generatedata.py ../Data/Datasets/filtered/test/ 64 64 ../Data/Datasets/keras/test 0 1