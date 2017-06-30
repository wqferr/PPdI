import os
import math
from sys import argv

from random import shuffle
from shutil import copyfile as cp

# get an argument from argv with a default value
def get_arg(n, default):
    if len(argv) > n:
        return argv[n]
    return default

# execute cp command and add prefix if required
def copy_img(lst, frm, to, rename):
    for entry in lst:
        name = os.path.basename(entry.name)
        if rename:
            new_name = os.path.join(frm, name)
            new_name = new_name.replace(os.sep, '_')
        else:
            new_name = name
        new_path = os.path.join(to, new_name)
        cp(entry.path, new_path)

# Args:
# 1 - The percent of images which will be used as the test dataset (default 20)
# 2 - The source directory of the images (default '.')
# 3 - The output root directory (default source directory)
# 4 - The training dataset subdirectory (default 'train')
# 5 - The test dataset subdirectory (default 'test')
# 6 - Boolean value, true if should prepend origin directory to file copy (default false)
if __name__ == '__main__':
    test_percent = float(get_arg(1, 20)) / 100
    src_dir = get_arg(2, '.')
    out_dir = get_arg(3, src_dir)
    train_subdir = get_arg(4, 'train')
    test_subdir = get_arg(5, 'test')
    rename = get_arg(6, 'false').strip().lower() == 'true'

    train_dir = os.path.join(out_dir, train_subdir)
    test_dir = os.path.join(out_dir, test_subdir)

	# create output directory if it doesn't exist
    os.makedirs(train_dir, 0o777, True)
    os.makedirs(test_dir, 0o777, True)

	# list all images in input directory
    images = []
    entries = os.scandir(src_dir)
    for entry in entries:
        if entry.is_file():
            images.append(entry)
    del entries

	# order them randomly
    shuffle(images)
	
	# n_test first will be part of the test set, and the remaining will be used for training
    n_test = math.ceil(len(images) * test_percent)
    train_images = images[n_test:]
    test_images = images[:n_test]
    
	# copy images to test and train directories
    copy_img(train_images, src_dir, train_dir, rename)
    copy_img(test_images, src_dir, test_dir, rename)
