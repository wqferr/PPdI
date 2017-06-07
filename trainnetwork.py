from keras.datasets import cifar10
from keras import utils
from keras import models
import numpy as np

vgg16_imgnet = models.load_model("../Data/vgg16_imgnet.h5")
vgg16_imgnet.summary()

images = np.load("../Data/Datasets/keras/test-images.npy", mmap_mode='r+')
labels = np.load("../Data/Datasets/keras/test-labels.npy", mmap_mode='r+')

vgg16_imgnet.fit(images, labels, epochs=30, batch_size=32)
vgg16_imgnet.summary()
vgg16_imgnet.save("../Data/vgg16_trained.h5")