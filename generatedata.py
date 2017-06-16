#Author: Eduardo Santos Carlos de Souza

#Usage:
#argv[1] = output image height
#argv[2] = output image width
#argv[3] = # of batches
#argv[4] = path to load input images
#argv[5] = path to save output arrays
#argv[6] = prefix for the arrays' filenames
#argv[7] = bool to apply random variations

from keras.preprocessing.image import ImageDataGenerator
from math import ceil
import numpy as np
import time
import sys
import os.path

out_shape = (224, 224)
dataset_rep_count = 10
in_path = os.path.join("..", "Data", "Datasets", "filtered")
out_path = os.path.join("..", "Data", "Datasets", "keras")
out_prefix = ""
rand = True
if (len(sys.argv) >= 3):
	out_shape = (int(sys.argv[1]), int(sys.argv[2]))
if (len(sys.argv) >= 4):
	dataset_rep_count = int(sys.argv[3])
if (len(sys.argv) >= 5):
	in_path = sys.argv[4]
if (len(sys.argv) >= 6):
	out_path = sys.argv[5]
if (len(sys.argv) >= 7):
	out_prefix = sys.argv[6] + "_"
if (len(sys.argv) >= 8):
	rand = (sys.argv[7] == 'true')

batch_size = 256
img_gen = ImageDataGenerator(fill_mode='constant', cval=0.0)
if (rand):
	img_gen = ImageDataGenerator(rotation_range=15.0, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, fill_mode='constant', cval=0.0, horizontal_flip=True, vertical_flip=False)
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
	np.save(os.path.join(out_path, out_prefix + "images"), x)
	np.save(os.path.join(out_path, out_prefix + "labels"), y)