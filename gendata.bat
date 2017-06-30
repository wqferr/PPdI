python3 generatevgg16.py
python3 sepdatasets.py
python3 generatedata.py ../Data/Datasets/separated/train/ 64 64 Data/Datasets/keras/train
python3 generatedata.py ../Data/Datasets/separated/test/ 64 64 Data/Datasets/keras/test 0 1