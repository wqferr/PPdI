from keras.datasets import cifar10
from keras import utils
from keras import models

vgg16_imgnet = models.load_model("vgg16_imgnet.h5")
vgg16_imgnet.summary()

(img_train, class_train), (img_test, class_test) = cifar10.load_data()
class_train = utils.np_utils.to_categorical(class_train, 10)

vgg16_imgnet.fit(img_train, class_train, epochs=5, batch_size=256)
vgg16_imgnet.summary()
vgg16_imgnet.save("vgg16_trained.h5")