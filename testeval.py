from keras.datasets import cifar10
from keras import utils
from keras import models
import numpy as np

vgg16_imgnet = models.load_model("vgg16_trained.h5")
vgg16_imgnet.summary()

(img_train, class_train), (img_test, class_test) = cifar10.load_data()
class_test = utils.np_utils.to_categorical(class_test, 10)

predictions = vgg16_imgnet.predict(img_test, batch_size=256)

test_size = 10000
pred_class = np.zeros(test_size)
right_class = np.zeros(test_size)
for i in range(test_size):
	pred_class[i] = np.argmax(predictions[i])
	right_class[i] = np.argmax(class_test[i])

print(right_class)
print(pred_class)

erros = np.count_nonzero(right_class - pred_class)
print("Errors: %d" % erros)
print("Accuracy: %lf" % ((test_size - erros)/test_size))