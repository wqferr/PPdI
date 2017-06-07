#Author: Eduardo Santos Carlos de Souza

#Usage:
#argv[1] = filename of model to train
#argv[2] = filename of numpy-array file of the images
#argv[3] = filename of numpy-array file of the labels
#argv[4] = filename to save the trained network

from keras import models
from os import path
import numpy as np
import sys

model_filename = "../Data/vgg16_imgnet.h5"
if (path.isfile("../Data/vgg16_trained.h5")):
	model_filename = "../Data/vgg16_trained.h5"
images_filename = "../Data/Datasets/keras/images.npy"
labels_filename = "../Data/Datasets/keras/labels.npy"
out_filename = "../Data/vgg16_trained.h5"
if (len(sys.argv) >= 2):
	model_filename = sys.argv[1]
if (len(sys.argv) >= 4):
	images_filename = sys.argv[2]
	labels_filename = sys.argv[3]
if (len(sys.argv) >= 5):
	out_filename = sys.argv[4]

images = np.load(images_filename, mmap_mode='r')
labels = np.load(labels_filename, mmap_mode='r')
cnn = models.load_model(model_filename)
cnn.summary()

cnn.fit(images, labels, epochs=30, batch_size=32)
cnn.summary()
cnn.save(out_filename)