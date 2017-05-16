#Author: Eduardo Santos Carlos de Souza

#Usage:
#argv[1] = path with images to standardized
#argv[2] = name of the folder to store the standardized images
#argv[3] = standardized height
#argv[4] = standardized width
#argv[5] = standardized image type

import cv2
import os
import sys

path = "../Data/Datasets/raw"
new_folder = "/standardized"
output_shape = (128, 128)
suffix = ".png"

file_tree = []
for dire, subdirs, files in os.walk(path):
	file_tree.append((dire, subdirs, files))

crop_size = len(path)
new_path = '/'.join(path.split('/')[:-1]) + new_folder
for dire, subdirs, files in file_tree:
	if os.path.exists(new_path + dire[crop_size:]) == False:
		os.mkdir(new_path + dire[crop_size:])

for dire, subdirs, files in file_tree:
	for filename in files:
		img = cv2.imread(dire + "/" + filename)
		if img != None:
			height, width, channels = img.shape
			img = cv2.resize(img, output_shape, interpolation=cv2.INTER_CUBIC)
			cv2.imwrite(new_path + dire[crop_size:] + "/" + ''.join(filename.split('.')[:-1]) + suffix, img)