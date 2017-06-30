#Author: Eduardo Santos Carlos de Souza

#Usage:
#argv[1] = path to the CNN
#argv[2] = array of the output classes separated by ','

from keras import models
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os.path
import sys

#Argumentos
model_filename = os.path.join("..", "Data", "CNN", "vgg16_imgnet.h5")
if (os.path.isfile(os.path.join("..", "Data", "CNN", "vgg16_fine_tuned_3.h5"))):
	model_filename = os.path.join("..", "Data", "CNN", "vgg16_fine_tuned_3.h5")
classes = ["Bycicle", "Car", "Dog", "Motorcycle", "No Parking", "Pedestrian", "Stop Light", "Stop Sign", "Toll", "Truck"]
if (len(sys.argv) >= 2):
	model_filename = sys.argv[1]
if (len(sys.argv) >= 3):
	classes = sys.argv[2].split(',')

cnn = models.load_model(model_filename)
cnn.summary()

#Design do gráfico de barras
points = np.linspace(0.5, len(classes)-0.5, len(classes))
fig = plt.figure(figsize=(5, 5))
fig.canvas.set_window_title("CNN Output")
sub_plt = fig.add_subplot(111)
sub_plt.set_title("CNN Output")
sub_plt.set_xlabel("Class")
sub_plt.set_ylabel("Value")
sub_plt.set_xticks(points)
sub_plt.set_xticklabels(classes, rotation='vertical', horizontalalignment='center')
bars = sub_plt.bar(points-0.4, np.zeros(len(classes)))
cam = cv2.VideoCapture(0)
fig.tight_layout()
img_shape = (cnn.get_layer(index=0).input_shape[1], cnn.get_layer(index=0).input_shape[2])

plt.show(block=False)
while True:
	#Obter Imagem da camera
	img = cam.read()[1]
	cv2.imshow("Imagem", img)

	#Modificar tamanho da imagem e passar pela rede
	img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), img_shape)
	img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
	pred = cnn.predict(img, batch_size=1)[0]
	print(classes[np.argmax(pred)])

	#Desenhar o grafico
	bars.remove()
	bars = sub_plt.bar(points-0.4, pred)
	fig.canvas.draw()

	#Sair caso a tecla ESC seja apertada
	ch = cv2.waitKey(1)
	if ch == 27:
		break

cv2.destroyAllWindows()