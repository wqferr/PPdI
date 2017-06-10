#Author: Eduardo Santos Carlos de Souza

#Usage:
#argv[1] = number of epochs to train
#argv[2] = filepath of folder with numpy-array files of the images and labels for testing and training
#Its going to look inside the directory for files names train_images.npy, train_labels.npy, test_images.npy, test_labels.npy, 
#argv[3] = filename of model to train
#argv[3] = filename to save the trained network

from keras import models, metrics
from os import path
import numpy as np
import sys
import time

epochs = 30
model_filename = path.join("..", "Data", "vgg16_imgnet.h5")
if (path.isfile(path.join("..", "Data", "vgg16_trained.h5"))):
	model_filename = path.join("..", "Data", "vgg16_trained.h5")
arrays_filepath = path.join("..", "Data", "Datasets", "keras")
out_filename = path.join("..", "Data", "vgg16_trained.h5")
if (len(sys.argv) >= 2):
	epochs = int(sys.argv[1])
if (len(sys.argv) >= 3):
	arrays_filepath = sys.argv[2]
if (len(sys.argv) >= 4):
	model_filename = sys.argv[3]
if (len(sys.argv) >= 5):
	out_filename = sys.argv[4]

train_images = np.load(path.join(arrays_filepath,"train_images.npy"), mmap_mode='r')
train_labels = np.load(path.join(arrays_filepath,"train_labels.npy"), mmap_mode='r')
test_images = np.load(path.join(arrays_filepath,"test_images.npy"), mmap_mode='r')
test_labels = np.load(path.join(arrays_filepath,"test_labels.npy"), mmap_mode='r')
cnn = models.load_model(model_filename)
cnn.summary()

batch_size = 256
for i in range(epochs):
	init_time = time.time()
	print("Epoch: %d" % i)
	train_score = cnn.fit(train_images, train_labels, epochs=1, batch_size=batch_size, verbose=False)
	test_score = cnn.evaluate(test_images, test_labels, batch_size=batch_size, verbose=False)
	print("Train Loss: %.4lf\nTrain Accuracy: %.4lf" % (train_score.history['loss'][0], train_score.history['acc'][0]))
	print("Test Loss: %.4lf\nTest Accuracy: %.4lf" % (test_score[0], test_score[1]))
	print("Time: %ds\n" % (time.time()-init_time))

cnn.summary()
cnn.save(out_filename)