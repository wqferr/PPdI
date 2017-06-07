#Author: Eduardo Santos Carlos de Souza

#Usage:
#argv[1] = standardized height
#argv[2] = standardized width
#argv[3] = # of batches
#argv[4] = path of input images
#argv[5] = path to store output arrays
#argv[6] = prefix for the array's filename

from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import time
import sys

out_shape = (224, 224)
batch_count = 5
in_path = "../Data/Datasets/raw"
out_path = "../Data/Datasets/keras/test-"
if (len(sys.argv) >= 3):
	out_shape = (int(sys.argv[1]), int(sys.argv[2]))
if (len(sys.argv) >= 4):
	batch_count = int(sys.argv[3])
if (len(sys.argv) >= 5):
	in_path = sys.argv[4]
if (len(sys.argv) >= 6):
	out_path = sys.argv[5]
if (len(sys.argv) >= 7):
	out_path = path + sys.argv[6] + "-"

img_gen = ImageDataGenerator(rotation_range=30.0, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, fill_mode='constant', cval=0.0, horizontal_flip=True, vertical_flip=False)
img_flow = img_gen.flow_from_directory(in_path, target_size=out_shape, class_mode='categorical', batch_size=256, shuffle=True, seed=int(time.time()))

if (batch_count > 0):
	x, y = img_flow.next()
	for i in range(0, batch_count-1):
		aux_x, aux_y = img_flow.next()
		x = np.concatenate((x, aux_x))
		y = np.concatenate((y, aux_y))

	np.save(out_path + "images", x)
	np.save(out_path + "labels", y)