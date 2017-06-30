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
def copy_img(lst, frm, to):
	for entry in lst:
		name = os.path.basename(entry.name)
		new_path = os.path.join(to, name)
		cp(entry.path, new_path)

def copy_dataset(src_dir_entry, out_dir):
	dataset = src_dir_entry.name
	src_dir = src_dir_entry.path
	test_dir = os.path.join(out_dir, 'test', dataset)
	train_dir = os.path.join(out_dir, 'train', dataset)
	
	os.makedirs(train_dir, 0o777, True)
	os.makedirs(test_dir, 0o777, True)
	
	# find every file in the subdirectory
	images = []
	entries = os.scandir(src_dir)
	for entry in entries:
		if entry.is_file():
			images.append(entry)
	
	# order them randomly
	shuffle(images)
	
	# n_test first will be part of the test set, and the remaining will be used for training
	n_test = math.ceil(len(images) * test_percent)
	train_images = images[n_test:]
	test_images = images[:n_test]
	
	# copy images to test and train directories
	copy_img(train_images, src_dir, train_dir)
	copy_img(test_images, src_dir, test_dir)

	
# Args:
# 1 - The percent of images which will be used as the test dataset (default 20)
# 2 - The source directory of the images (default '../Data/Datasets/filtered')
# 3 - The output root directory (default ../Data/Datasets/separated)
if __name__ == '__main__':
	test_percent = float(get_arg(1, 20)) / 100
	src_dir = get_arg(2, os.path.join('..', 'Data', 'Datasets', 'filtered'))
	out_dir = get_arg(3, os.path.join('..', 'Data', 'Datasets', 'separated'))

	# treat every subdirectory of src_dir as a class and copy it to separated/
	entries = os.scandir(src_dir)
	for entry in entries:
		if entry.is_dir():
			copy_dataset(entry, out_dir)