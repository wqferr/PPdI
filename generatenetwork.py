#Author: Eduardo Santos Carlos de Souza

#Usage:
#argv[1] = input image height
#argv[2] = input image width
#argv[3] = input image # of channels
#argv[4] = output # of classes
#argv[5] = filename to store the generated Model

from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense
from keras.models import Model
import sys

#Variaveis de entrada e saida da rede
in_shape = (224, 224, 3)
n_classes = 1000
filename = "../Data/vgg16_imgnet.h5"
if (len(sys.argv) >= 4):
	in_shape = (int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
if (len(sys.argv) >= 5):
	n_classes = int(sys.argv[4])
if (len(sys.argv) >= 6):
	filename = sys.argv[5]

#Baixar o modelo treinado na ImageNet sem as camadas de input e output, com max pooling; i.e Baixar camadas convolucionais
vgg16_imgnet = VGG16(weights='imagenet', include_top=False, input_tensor=None, input_shape=None, pooling='softmax')
vgg16_imgnet.summary()

#Congelar as camadas convolucionas
#for layer in vgg16_imgnet.layers:
#	layer.trainable = False

#Adicionar camada de Input
input_layer = Input(shape=in_shape, name='input')
vgg16_imgnet_tensor = vgg16_imgnet(input_layer)

#Adicionar camadas fully-connected
vgg16_imgnet_tensor = Flatten(name='flatten')(vgg16_imgnet_tensor)
vgg16_imgnet_tensor = Dense(4096, activation='relu', name='fullyconnected_1')(vgg16_imgnet_tensor)
vgg16_imgnet_tensor = Dense(4096, activation='relu', name='fullyconnected_2')(vgg16_imgnet_tensor)
vgg16_imgnet_tensor = Dense(n_classes, activation='softmax', name='classifier')(vgg16_imgnet_tensor)

#Gerar modelo
new_vgg16_imgnet = Model(input_layer, vgg16_imgnet_tensor)
new_vgg16_imgnet.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
new_vgg16_imgnet.summary()

#Salvar modelo
new_vgg16_imgnet.save(filename)