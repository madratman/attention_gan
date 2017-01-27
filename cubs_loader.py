import os, cv2
import numpy as np
from pprint import pprint
if not (os.uname()[1]=='ratneshmadaan-Inspiron-N5010'):
	from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# the original cubs dataset has each type of bird's image's in a diff subfolder. Let's first put these into a single train and test folder
# (the split is specified in train_test_split.txt), so that we can use keras' data preprocessors

# meta list has each element in form [idx, flag]. idx goes till 11788, flag=1 it's in training set. 

class CUB_Loader():
	def __init__(self, flag_split_train_test=0, **kwargs):
		self.repo_path = os.path.dirname(os.path.realpath(__file__))
		self.cubs_root = os.path.join(self.repo_path, 'data/CUB_200_2011')
		self.image_dir = os.path.join(self.cubs_root, 'images')
		self.image_id_mapping_file = os.path.join(self.cubs_root, 'images.txt') # this file provides idx for each image. they are NOT sorted alphabetically 
		self.train_test_flag_file = os.path.join(self.cubs_root, 'train_test_split.txt') # this tells if each image specified by the index is in training set(1) or test set(0)
		self.image_class_labels_file = os.path.join(self.cubs_root, 'image_class_labels_file.txt')
		self.bounding_box_file = os.path.join(self.cubs_root, 'bounding_boxes.txt')
		self.segmentation_label_dir = 'data/segmentations'
		self.flag_split_train_test = flag_split_train_test
		# http://stackoverflow.com/questions/1747817/create-a-dictionary-with-list-comprehension-in-python
		# d = dict((key, value) for (key, value) in iterable)
		self.image_names = {}
		with open(self.image_id_mapping_file) as f:
			self.image_names = dict( (int(line.split()[0]), line.split()[1]) for line in f.readlines())

		# this is like : 8018: ['104.0', '95.0', '212.0', '154.0']
		self.bboxes = {}
		with open(self.bounding_box_file) as f:
			self.bboxes = dict( (int(line.split()[0]), line.split()[1:]) for line in f.readlines() )

		# train has value = 1
		with open(self.train_test_flag_file) as f:
			self.train_or_test = dict( (int(line.split()[0]), int(line.split()[1])) for line in f.readlines() )

	def _split_into_train_and_test(self):
		if self.flag_split_train_test==1:
			train_idx_list = []
			test_idx_list = []

			with open(self.train_test_flag_file) as file:
				for each_line in file:
					curr_list = [int(each_number.strip()) for each_number in each_line.split(' ')]
					# print curr_list
					if curr_list[1]==1:
						train_idx_list.append(curr_list[0])
					else:
						test_idx_list.append(curr_list[0]) 

			print "len(train_idx_list) : ", len(train_idx_list) # 5994
			print "len(test_idx_list) : ", len(test_idx_list) # 5794

			image_id_dict = {}
			with open(image_id_mapping_file) as file:
				for each_line in file:
					curr_list = [element.strip() for element in each_line.split(' ')]
					image_id_dict[int(curr_list[0])] = curr_list[1]

			training_image_paths = [image_id_dict[train_id] for train_id in train_idx_list]
			testing_image_paths = [image_id_dict[test_id] for test_id in test_idx_list]
			# pprint(training_image_paths)
			# pprint(testing_image_paths)

			# move training images into one dir
			if not os.path.exists('data/train'):
				os.makedirs('data/train')
			if not os.path.exists('data/test'):
				os.makedirs('data/test')

			for idx in range(len(training_image_paths)):
				directory_name = training_image_paths[idx].split('/')[0]
				image_basename = training_image_paths[idx].split('/')[1] 
				
				if not os.path.exists('data/train/'+directory_name):
					os.makedirs('data/train/'+directory_name)

				new_filename = os.path.join('data/train/'+directory_name, image_basename)
				old_filename = os.path.join(self.image_dir, training_image_paths[idx])
				
				# move file
				os.rename(old_filename, new_filename)

			for idx in range(len(testing_image_paths)):
				directory_name = testing_image_paths[idx].split('/')[0]
				image_basename = testing_image_paths[idx].split('/')[1] 
				
				if not os.path.exists('data/test/'+directory_name):
					os.makedirs('data/test/'+directory_name)

				new_filename = os.path.join('data/test/'+directory_name, image_basename)
				old_filename = os.path.join(self.image_dir, testing_image_paths[idx])
				
				# move file
				os.rename(old_filename, new_filename)

		elif self.flag_split_train_test==0:
			raise ValueError('self.flag_split_train_test==0. Not doing anything')

	def _get_data_statistics(self):
		class_list = sorted(os.listdir(self.image_dir))
		no_of_images_arr = np.empty(len(class_list))
		for class_idx in range(len(class_list)):
			curr_dir = os.path.join(self.image_dir, class_list[class_idx])
			no_of_files_curr = len([file for file in os.listdir(curr_dir) if os.path.isfile(os.path.join(curr_dir, file))])
			print "{} : {} : no of images :: {} ".format(class_idx, curr_dir, no_of_files_curr)
			no_of_images_arr[class_idx] = no_of_files_curr
		print "mean number of images : {}".format(no_of_images_arr.mean())
		print "stdeviation of images : {}".format(no_of_images_arr.std())

	def _get_image_sizes(self):
		class_list = sorted(os.listdir(self.image_dir))
		no_of_images_arr = np.empty(len(class_list))
		for class_idx in range(len(class_list)):
			print class_idx
			curr_dir = os.path.join(self.image_dir, class_list[class_idx])
			list_of_images = sorted(os.listdir(curr_dir))
			for image_idx in range(len(list_of_images)):
				# print list_of_images[image_idx]
				curr_image = cv2.imread(os.path.join(curr_dir, list_of_images[image_idx]))

if __name__=='__main__':
	CUBS = CUB_Loader(flag_split_train_test=0)
	# CUBS._split_into_train_and_test()
	CUBS._get_data_statistics()
