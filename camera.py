from keras import models
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import cv2

model_filename = "../Data/CNN/vgg16_fine_tuned_3.h5"
cnn = models.load_model(model_filename)
cnn.summary()

classes = ["Bycicle", "Car", "Dog", "Motorcycle", "No Parking", "Pedestrian", "Stop Light", "Stop Sign", "Toll", "Truck"]

points = range(len(classes))
fig = plt.figure()
fig.canvas.set_window_title("CNN Output")
sub_plt = fig.add_subplot(111)
sub_plt.set_title("CNN Output")
sub_plt.set_xlabel("Class")
sub_plt.set_ylabel("Value")
sub_plt.set_xticks(points)
sub_plt.set_xticklabels(classes, rotation='vertical', horizontalalignment='center')
plt.show(block=False)

bars = sub_plt.bar(points, np.zeros(10))
cam = cv2.VideoCapture(1)
while True:
	img = cam.read()[1]
	cv2.imshow("Imagem", img)

	img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (64, 64))
	img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
	#img = image.img_to_array(image.load_img("/home/wheatley/Desktop/a.jpg", target_size=(64, 64)))
	#img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
	pred = cnn.predict(img, batch_size=1)[0]
	print(classes[np.argmax(pred)])

	bars.remove()
	bars = sub_plt.bar(points, pred)
	fig.canvas.draw()

	ch = cv2.waitKey(1)
	if ch == 27:
		break

cv2.destroyAllWindows()