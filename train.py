from sys import argv

import keras

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


# argv = ['train.py', 'class', rotation_range, width_shift_range, height_shift_range,
#           rescale, shear_range, zoom_range]
argv = argv[1:]

# argv = ['class', rotation_range, width_shift_range, height_shift_range,
#           rescale, shear_range, zoom_range]

# Default args
arg = [150, 45, 0.2, 0.2, 1./255, 0.2, 0.2]

arg = [float(x) for x in argv] + arg[len(argv):]


model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(3, arg[0], arg[0])))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])

batch_size = 16

train_datagen = ImageDataGenerator(
        rotation_range=arg[1],
        width_shift_range=arg[2],
        height_shift_range=arg[3],
        rescale=arg[4],
        shear_range=arg[5],
        zoom_range=arg[6],
        horizontal_flip=True,
        fill_mode='nearest',
        save_to_dir='out')

train_generator = train_datagen.flow_from_directory(
        'data',
        target_size=(arg[0], arg[0]),
        batch_size=batch_size,
        class_mode='binary')

test_datagen = ImageDataGenerator(
        rescale=1./255,
        save_to_dir='out')

validation_generator = test_datagen.flow_from_directory(
        'data',
        target_size=(arg[0], arg[0]),
        batch_size=batch_size,
        class_mode='binary')

model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)
