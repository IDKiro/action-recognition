from __future__ import print_function, division
import os
import shutil
import sys

def class_process(dir_path, valid_dir_path, class_name):
	class_path = os.path.join(dir_path, class_name)
	if not os.path.isdir(class_path):
		return

	valid_class_path = os.path.join(valid_dir_path, class_name)
	if not os.path.exists(valid_class_path):
		os.mkdir(valid_class_path)

	for i, (file_name) in enumerate(os.listdir(class_path)):
		name, ext = os.path.splitext(file_name)
		train_directory_path = os.path.join(class_path, name)
		valid_directory_path = os.path.join(valid_class_path, name)

		if i % 5 == 0:
			shutil.move(train_directory_path, valid_directory_path)


if __name__=="__main__":
	dir_path = sys.argv[1]
	valid_dir_path = sys.argv[2]
	for class_name in os.listdir(dir_path):
		class_process(dir_path, valid_dir_path, class_name)
