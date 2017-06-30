#Author: Eduardo Santos Carlos de Souza

#Usage:
#argv[1] = input image height
#argv[2] = input image width
#argv[3] = output # of classes
#argv[4] = filename to store the generated Model
#argv[5] = bool to fine tune

from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense
from keras.models import Model
import os.path
import gc
import sys

#Variaveis de entrada e saida da rede
in_shape = (64, 64, 3)
n_classes = 10
model_filename = os.path.join("..", "Data", "CNN", "vgg16_imgnet.h5")
fine_tune = False
if (len(sys.argv) >= 3):
	in_shape = (int(sys.argv[1]), int(sys.argv[2]), 3)
if (len(sys.argv) >= 4):
	n_classes = int(sys.argv[3])
if (len(sys.argv) >= 5):
	model_filename = sys.argv[4]
if (len(sys.argv) >= 6):
	fine_tune = (sys.argv[5] == 'true')

#Baixar o modelo treinado na ImageNet sem as camadas de input e output, com max pooling; i.e Baixar camadas convolucionais
vgg16_imgnet = VGG16(weights='imagenet', include_top=False, input_shape=in_shape)
vgg16_imgnet.summary()

#Congelar as camadas convolucionas
if (not fine_tune):
	for layer in vgg16_imgnet.layers:
		layer.trainable = False

#Adicionar camadas fully-connected
new_tensor = Flatten(name='flatten')(vgg16_imgnet.output)
new_tensor = Dense(4096, activation='relu', name='fullyconnected_1')(new_tensor)
new_tensor = Dense(4096, activation='relu', name='fullyconnected_2')(new_tensor)
new_tensor = Dense(n_classes, activation='softmax', name='classifier')(new_tensor)

#Gerar modelo
new_vgg16_imgnet = Model(vgg16_imgnet.input, new_tensor)
new_vgg16_imgnet.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
new_vgg16_imgnet.summary()

#Salvar modelo
new_vgg16_imgnet.save(model_filename)
gc.collect()