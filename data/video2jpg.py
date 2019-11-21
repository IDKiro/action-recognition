from __future__ import print_function, division
import os
import sys
import subprocess
import shutil

def class_process(dir_path, dst_dir_path, class_name, maxSize=1024):
	class_path = os.path.join(dir_path, class_name)
	if not os.path.isdir(class_path):
		return

	dst_class_path = os.path.join(dst_dir_path, class_name)
	if not os.path.exists(dst_class_path):
		os.makedirs(dst_class_path)

	for file_name in os.listdir(class_path):
		if '.avi' not in file_name:
			continue
		name, ext = os.path.splitext(file_name)
		dst_directory_path = os.path.join(dst_class_path, name)

		video_file_path = os.path.join(class_path, file_name)

		# skip large files
		# if os.path.getsize(video_file_path) > maxSize * 1000:
		#	continue

		try:
			if os.path.exists(dst_directory_path):
				if not os.path.exists(os.path.join(dst_directory_path, 'image_00001.jpg')):
					subprocess.call('rm -r \"{}\"'.format(dst_directory_path), shell=True)
					print('remove {}'.format(dst_directory_path))
					os.makedirs(dst_directory_path)
				else:
					continue
			else:
				os.makedirs(dst_directory_path)
		except:
			print(dst_directory_path)
			continue
	
		# cmd = 'ffmpeg -i \"{}\" -vf scale=-1:240 \"{}/image_%05d.jpg\"'.format(video_file_path, dst_directory_path)
		cmd = 'ffmpeg -i \"{}\" -qscale:v 2 \"{}/image_%05d.jpg\"'.format(video_file_path, dst_directory_path)
		
		print(cmd)
		subprocess.call(cmd, shell=True)
		print('\n')


def class_move(dir_path, valid_dir_path, class_name):
	class_path = os.path.join(dir_path, class_name)
	if not os.path.isdir(class_path):
		return

	valid_class_path = os.path.join(valid_dir_path, class_name)
	if not os.path.exists(valid_class_path):
		os.makedirs(valid_class_path)

	for i, (file_name) in enumerate(os.listdir(class_path)):
		name, ext = os.path.splitext(file_name)
		train_directory_path = os.path.join(class_path, name)
		valid_directory_path = os.path.join(valid_class_path, name)

		if i % 10 == 0:
			shutil.move(train_directory_path, valid_directory_path)


if __name__=="__main__":
	dir_path = './data/video/'
	dst_dir_path = './data/train/'
	valid_dir_path = './data/valid/'

	for class_name in os.listdir(dir_path):
		class_process(dir_path, dst_dir_path, class_name)
		class_move(dst_dir_path, valid_dir_path, class_name)
