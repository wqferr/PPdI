#Author: Eduardo Santos Carlos de Souza

#Usage:
#argv[1] = number of epochs to train
#argv[2] = filepath of folder with numpy-array files of the images and labels for testing and training
#Its going to look inside the directory for files names train_images.npy, train_labels.npy, test_images.npy, test_labels.npy
#argv[3] = filename of model to train
#argv[4] = bool to evaluate on test data each epoch
#argv[5] = filename to save the trained network

from keras import backend, models, metrics
import os.path
import numpy as np
import sys
import time

#Argumentos
epochs = 30
model_filename = os.path.join("..", "Data", "CNN", "vgg16_imgnet.h5")
if (os.path.isfile(os.path.join("..", "Data", "CNN", "vgg16_fine_tuned_3.h5"))):
	model_filename = os.path.join("..", "Data", "CNN", "vgg16_fine_tuned_3.h5")
arrays_filepath = os.path.join("..", "Data", "Datasets", "keras")
out_filename = os.path.join("..", "Data", "CNN", "vgg16_trained.h5")
evalu = False
if (len(sys.argv) >= 2):
	epochs = int(sys.argv[1])
if (len(sys.argv) >= 3):
	arrays_filepath = sys.argv[2]
if (len(sys.argv) >= 4):
	model_filename = sys.argv[3]
if (len(sys.argv) >= 5):
	evalu = (sys.argv[4] == 'true')
if (len(sys.argv) >= 6):
	out_filename = sys.argv[5]

#Carregar imagens, labels e modelos
train_images = np.load(os.path.join(arrays_filepath,"train_images.npy"), mmap_mode='r')
train_labels = np.load(os.path.join(arrays_filepath,"train_labels.npy"), mmap_mode='r')
test_images = np.load(os.path.join(arrays_filepath,"test_images.npy"), mmap_mode='r')
test_labels = np.load(os.path.join(arrays_filepath,"test_labels.npy"), mmap_mode='r')
cnn = models.load_model(model_filename)
cnn.summary()

#Treinar pelo numero de epochs necessário
batch_size = 128
for i in range(epochs):
	init_time = time.time()
	print("Epoch: %d" % (i+1))
	train_score = cnn.fit(train_images, train_labels, epochs=1, batch_size=batch_size, verbose=False)
	print("Train Loss: %.4lf\nTrain Accuracy: %.4lf" % (train_score.history['loss'][0], train_score.history['acc'][0]))
	#Avaliar a rede caso requerido
	if (evalu):
		test_score = cnn.evaluate(test_images, test_labels, batch_size=batch_size, verbose=False)
		print("Test Loss: %.4lf\nTest Accuracy: %.4lf" % (test_score[0], test_score[1]))
	#Tempo total da epoch
	print("Time: %ds\n" % (time.time()-init_time))

cnn.summary()
cnn.save(out_filename)

backend.clear_session()