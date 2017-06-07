from keras import models
import matplotlib.pyplot as plt
import numpy as np
import cv2

model_filename = "../Data/vgg16_imgnet.h5"
cnn = models.load_model(model_filename)
cnn.summary()

classes = ["Carro", "Moto", "Pedestre", "Pare", "Semaforo"]
cam = cv2.VideoCapture(0)
while True:
	img = cam.read()[1]
	cv2.imshow("Imagem", img)

	img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (50, 50))
	img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
	pred = cnn.predict(img, batch_size=1)
	print(classes[np.argmax(pred)])

	ch = cv2.waitKey(1)
	if ch == 27:
		break

#testimg = cv2.cvtColor(cv2.imread("1.jpg"), cv2.COLOR_BGR2RGB)
#plt.imshow(img)
#plt.show()
#testimg2 = image.img_to_array(image.load_img("1.jpg"))
#plt.imshow(img2)
#plt.show()
#res = img-img2
#plt.imshow(res)
#plt.show()
#print(res)

#cv2.imshow("hue", img)
#cv2.waitKey(1)

cv2.destroyAllWindows()