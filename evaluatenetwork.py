#Author: Eduardo Santos Carlos de Souza

#Usage:
#argv[1] = filename of model to evaluate
#argv[2] = filename of numpy-array file of the images
#argv[3] = filename of numpy-array file of the labels
#argv[4] = filename to save the predictions' array

from keras import models
from os import path
import numpy as np
import sys

model_filename = "../Data/vgg16_imgnet.h5"
if (path.isfile("../Data/vgg16_trained.h5")):
	model_filename = "../Data/vgg16_trained.h5"
images_filename = "../Data/Datasets/keras/images.npy"
labels_filename = "../Data/Datasets/keras/labels.npy"
out_filename =  "../Data/predictions.npy"
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

predictions = cnn.predict(images, batch_size=256)
np.save(out_filename, predictions)

pred_class = np.zeros(labels.shape[0])
right_class = np.zeros(labels.shape[0])
for i in range(labels.shape[0]):
	pred_class[i] = np.argmax(predictions[i])
	right_class[i] = np.argmax(labels[i])

print(right_class)
print(pred_class)

erros = np.count_nonzero(right_class - pred_class)
print("Errors: %d" % erros)
print("Accuracy: %lf" % ((labels.shape[0] - erros)/labels.shape[0]))