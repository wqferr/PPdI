#Author: Eduardo Santos Carlos de Souza

#Usage:
#argv[1] = output image height
#argv[2] = output image width
#argv[3] = output image # channels
#argv[4] = # of batches
#argv[5] = path to load input images
#argv[6] = path to save output arrays
#argv[7] = prefix for the arrays' filenames

from keras.preprocessing.image import ImageDataGenerator
from math import ceil
import numpy as np
import time
import sys

out_shape = (224, 224)
color = 'rgb'
dataset_rep_count = 20
in_path = "../Data/Datasets/raw"
out_path = "../Data/Datasets/keras/"
if (len(sys.argv) >= 4):
	out_shape = (int(sys.argv[1]), int(sys.argv[2]))
	if (int(sys.argv[3])):
		color = 'grayscale'
if (len(sys.argv) >= 5):
	dataset_rep_count = int(sys.argv[4])
if (len(sys.argv) >= 6):
	in_path = sys.argv[5]
if (len(sys.argv) >= 7):
	out_path = sys.argv[6]
if (len(sys.argv) >= 8):
	out_path = out_path + sys.argv[7] + "-"

batch_size = 256
img_gen = ImageDataGenerator(rotation_range=30.0, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, fill_mode='constant', cval=0.0, horizontal_flip=True, vertical_flip=False)
img_flow = img_gen.flow_from_directory(in_path, target_size=out_shape, class_mode='categorical', batch_size=batch_size, shuffle=False, seed=int(time.time()))

if (dataset_rep_count > 0):
	x, y = img_flow.next()
	limit = (dataset_rep_count * ceil(img_flow.samples/batch_size))-1
	for i in range(0, limit):
		aux_x, aux_y = img_flow.next()
		x = np.concatenate((x, aux_x))
		y = np.concatenate((y, aux_y))

	print("Images array: ", end="")
	print(x.shape)
	print("Labels array: ", end="")
	print(y.shape)
	np.save(out_path + "images", x)
	np.save(out_path + "labels", y)