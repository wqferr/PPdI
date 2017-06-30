#Author: Eduardo Santos Carlos de Souza

#Usage:
#argv[1] = filename of numpy-array file of the images
#argv[2] = filename of numpy-array file of the labels
#argv[3] = filename of model to evaluate

from keras import models
import os.path
import numpy as np
import sys

#Argumentos
model_filename = os.path.join("..", "Data", "CNN", "vgg16_trained.h5")
images_filename = os.path.join("..", "Data", "Datasets", "keras", "test_images.npy")
labels_filename = os.path.join("..", "Data", "Datasets", "keras", "test_labels.npy")
if (len(sys.argv) >= 3):
	images_filename = sys.argv[1]
	labels_filename = sys.argv[2]
if (len(sys.argv) >= 4):
	model_filename = sys.argv[3]

#Carregar modelo e imagens
images = np.load(images_filename, mmap_mode='r')
labels = np.load(labels_filename, mmap_mode='r')
cnn = models.load_model(model_filename)
cnn.summary()

#Avaliar rede
batch_size = 256
score = cnn.evaluate(images, labels, batch_size=batch_size)
print("Loss: %.4lf\nAccuracy: %.4lf" % (score[0], score[1]))