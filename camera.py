from keras import models
from os import path
import numpy as np
import cv2

model_filename = "../Data/vgg16_imgnet.h5"
if (path.isfile("../Data/vgg16_trained.h5")):
	model_filename = "../Data/vgg16_trained.h5"
cnn = models.load_model(model_filename)
cnn.summary()

classes = ["Bicicleta", "Cachorro", "Carro", "Moto", "Proibido Estacionar", "Pedestre", "Pare", "Pedagio", "Semaforo", "Caminhao"]
cam = cv2.VideoCapture(0)
while True:
	img = cam.read()[1]
	cv2.imshow("Imagem", img)

	img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (64, 64))
	img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
	pred = cnn.predict(img, batch_size=1)
	print(classes[np.argmax(pred)])

	ch = cv2.waitKey(1)
	if ch == 27:
		break

cv2.destroyAllWindows()