#Author: Eduardo Santos Carlos de Souza

#Usage:
#argv[1] = path to load input images
#argv[2] = output image height
#argv[3] = output image width
#argv[4] = path to save output arrays
#argv[5] = bool to apply random variations
#argv[6] = # of batches

from keras.preprocessing.image import ImageDataGenerator
from math import ceil
import numpy as np
import time
import sys
import os.path

#Argumentos
in_path = os.path.join("..", "Data", "Datasets", "separated", "train")
out_shape = (224, 224)
out_path = os.path.join("..", "Data", "Datasets", "keras", "train")
rand = True
dataset_rep_count = 10
if (len(sys.argv) >= 2):
	in_path = sys.argv[1]
if (len(sys.argv) >= 4):
	out_shape = (int(sys.argv[2]), int(sys.argv[3]))
if (len(sys.argv) >= 5):
	out_path = sys.argv[4]
if (len(sys.argv) >= 6):
	rand = (sys.argv[5] == 'true')
if (len(sys.argv) >= 7):
	dataset_rep_count = int(sys.argv[6])

#Criar um objeto que cria um iterador que percorre e classifica as imagens
#baseado na estrutura de pastas
batch_size = 256
img_gen = ImageDataGenerator(fill_mode='constant', cval=0.0)
if (rand):
	img_gen = ImageDataGenerator(rotation_range=15.0, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, fill_mode='constant', cval=0.0, horizontal_flip=True, vertical_flip=False)
img_flow = img_gen.flow_from_directory(in_path, target_size=out_shape, class_mode='categorical', batch_size=batch_size, shuffle=False, seed=int(time.time()))

#Gerar vetores finais com todas as imagens e labels
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
	np.save(out_path + "_images", x)
	np.save(out_path + "_labels", y)