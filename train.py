from sys import argv

import keras

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# argv = ['train.py', 'class', rotation_range, width_shift_range, height_shift_range,
#           rescale, shear_range, zoom_range]
argv = argv[1:]

# argv = ['class', rotation_range, width_shift_range, height_shift_range,
#           rescale, shear_range, zoom_range]

# Default args
arg = [45, 0.2, 0.2, 1./255, 0.2, 0.2]

arg = [float(x) for x in argv[1:]] + arg[len(argv)-1:]

datagen = ImageDataGenerator(
        rotation_range=arg[0],
        width_shift_range=arg[1],
        height_shift_range=arg[2],
        rescale=arg[3],
        shear_range=arg[4],
        zoom_range=arg[5],
        horizontal_flip=True,
        fill_mode='nearest')

img = load_img('data/{}/0.jpg'.format(argv[0]))
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1, save_to_dir='out', save_prefix=argv[0], save_format='jpeg'):
    i += 1
    if i > 20:
        break;
