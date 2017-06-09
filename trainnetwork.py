#Author: Eduardo Santos Carlos de Souza

#Usage:
#argv[1] = number of epochs to train
#argv[2] = filepath of folder with numpy-array files of the images and labels for testing and training
#argv[3] = filename of model to train
#argv[3] = filename to save the trained network

from keras import models
from os import path
import numpy as np
import sys

epochs = 30
model_filename = "../Data/vgg16_imgnet.h5"
if (path.isfile("../Data/vgg16_trained.h5")):
	model_filename = "../Data/vgg16_trained.h5"
arrays_filepath = "../Data/Datasets/keras/"
out_filename = "../Data/vgg16_trained.h5"
if (len(sys.argv) >= 2):
	epochs = int(sys.argv[1])
if (len(sys.argv) >= 3):
	arrays_filepath = sys.argv[2]
if (len(sys.argv) >= 4):
	model_filename = sys.argv[3]
if (len(sys.argv) >= 5):
	out_filename = sys.argv[4]

train_images = np.load(arrays_filepath+"train_images.npy", mmap_mode='r')
train_labels = np.load(labels_filename+"train_labels.npy", mmap_mode='r')
test_images = np.load(images_filename+"test_images.npy", mmap_mode='r')
test_labels = np.load(labels_filename+"test_labels.npy", mmap_mode='r')
cnn = models.load_model(model_filename)
cnn.summary()

#batch_size = 32
image_count = test_labels.shape[0]

loss = 0
accu = 0
predictions = np.zeros(test_labels.shape)
pred_class = np.zeros(image_count)
right_class = np.zeros(image_count)
for i in range(epochs):
	print("Epoch: %d" % i)
	cnn.fit(train_images, train_labels, epochs=1, batch_size=32, verbose=False)
	loss = cnn.evaluate(train_images, train_labels, batch_size=256)
	predictions = cnn.predict(test_images, batch_size=256)

	for j in range(image_count):
		pred_class[i] = np.argmax(predictions[i])
		right_class[i] = np.argmax(test_labels[i])

	accu = (image_count-np.count_nonzero(right_class - pred_class))/image_count
	print("Loss %.4lf\nAccuracy: %.4lf" % loss, accu)

cnn.summary()
cnn.save(out_filename)