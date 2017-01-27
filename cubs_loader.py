import os, cv2
import numpy as np
from pprint import pprint
if not (os.uname()[1]=='ratneshmadaan-Inspiron-N5010'):
	from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image

# the original cubs dataset has each type of bird's image's in a diff subfolder. Let's first put these into a single train and test folder
# (the split is specified in train_test_split.txt), so that we can use keras' data preprocessors

# meta list has each element in form [idx, flag]. idx goes till 11788, flag=1 it's in training set. 

class CUB_Loader():
	def __init__(self, flag_split_train_test=0, **kwargs):
		self.repo_path = os.path.dirname(os.path.realpath(__file__))
		self.cubs_root = os.path.join(self.repo_path, 'data/CUB_200_2011')
		self.image_dir = os.path.join(self.cubs_root, 'images')
		self.train_dir = os.path.join(self.repo_path, 'data/train')
		self.test_dir = os.path.join(self.repo_path, 'data/test')
		self.image_id_mapping_file = os.path.join(self.cubs_root, 'images.txt') # this file provides idx for each image. they are NOT sorted alphabetically 
		self.train_test_flag_file = os.path.join(self.cubs_root, 'train_test_split.txt') # this tells if each image specified by the index is in training set(1) or test set(0)
		self.image_class_labels_file = os.path.join(self.cubs_root, 'image_class_labels_file.txt')
		self.bounding_box_file = os.path.join(self.cubs_root, 'bounding_boxes.txt')
		self.segmentation_label_dir = 'data/segmentations'
		self.flag_split_train_test = flag_split_train_test
		self.split_done = False
		if os.path.exists(self.train_dir):
			self.split_done = True

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
		if self.split_done == True:
			raise ValueError("You've already split the data! Don't call me maybe?")

		if self.flag_split_train_test==0:
			raise ValueError('Fix the flag_split_train_test==0. Not doing anything')

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
			with open(self.image_id_mapping_file) as file:
				for each_line in file:
					curr_list = [element.strip() for element in each_line.split(' ')]
					image_id_dict[int(curr_list[0])] = curr_list[1]

			training_image_paths = [image_id_dict[train_id] for train_id in train_idx_list]
			testing_image_paths = [image_id_dict[test_id] for test_id in test_idx_list]
			# pprint(training_image_paths)
			# pprint(testing_image_paths)

			for idx in range(len(training_image_paths)):
				directory_name = training_image_paths[idx].split('/')[0]
				image_basename = training_image_paths[idx].split('/')[1] 
				
				if not os.path.exists(os.path.join(self.train_dir, directory_name)):
					os.makedirs(os.path.join(self.train_dir, directory_name))

				new_filename = os.path.join(self.train_dir, directory_name, image_basename)
				old_filename = os.path.join(self.image_dir, training_image_paths[idx])
				
				# move file
				os.rename(old_filename, new_filename)

			for idx in range(len(testing_image_paths)):
				directory_name = testing_image_paths[idx].split('/')[0]
				image_basename = testing_image_paths[idx].split('/')[1] 
				
				if not os.path.exists(os.path.join(self.test_dir, directory_name)):
					os.makedirs(os.path.join(self.test_dir, directory_name))

				new_filename = os.path.join(self.test_dir, directory_name, image_basename)
				old_filename = os.path.join(self.image_dir, testing_image_paths[idx])
				# print (old_filename, new_filename)
				
				# move file
				os.rename(old_filename, new_filename)
			
			import shutil
			self.split_done = True
			shutil.rmtree(self.image_dir)

	def _get_data_statistics(self, details_per_dir=0):
		class_list = sorted(os.listdir(self.image_dir))
		no_of_images_arr = np.empty(len(class_list))
		for class_idx in range(len(class_list)):
			curr_dir = os.path.join(self.image_dir, class_list[class_idx])
			no_of_files_curr = len([file for file in os.listdir(curr_dir) if os.path.isfile(os.path.join(curr_dir, file))])
			if details_per_dir:
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

	# load everything into ram and manually generate batches. will take ~10GB for 6000 224*224 images
	def _cubs_into_ram_all(self):
		if self.split_done == False:
			raise ValueError("You haven't split the data!!!")
		image_size = (224,224)
		no_of_training_images = 5994 # find data/train -type f | wc -l
		all_sub_dirs = sorted(os.listdir(self.train_dir))
		no_of_classes = len(all_sub_dirs)
		X_train = np.empty((no_of_training_images,3)+image_size, dtype='float32') #dtype?
		y_train = np.empty(no_of_training_images, dtype='int')
		# read, preprocess and dump all in single np array
		category_idx = 0
		image_idx = 0

		print "no_of_classes : {}".format(no_of_classes)
		print "no_of_training_images : {}".format(no_of_training_images)
		for each_dir in all_sub_dirs:
			curr_label = category_idx
			all_images_category = os.listdir(os.path.join(self.train_dir, each_dir))
			category_idx += 1
			#print category_idx
			for each_image in all_images_category:
				image_file = os.path.join(os.path.join(self.train_dir, each_dir, each_image))
				#print image_file
				img = Image.open(image_file)
				img = img.convert('RGB') # ensure 3 channel
				img = img.resize(image_size, resample=Image.NEAREST)
				img_array = np.asarray(img, dtype='float32')
				if len(img_array.shape) == 3:
					img_array = img_array.transpose(2, 0, 1)
				elif len(img_array.shape) == 2:
					img_array = img_array.reshape((1, x.shape[0], x.shape[1]))
				X_train[image_idx, ...] = img_array
				y_train[image_idx] = category_idx #Labels are one indexed
				image_idx += 1
				#print image_idx
				percent_done = float(image_idx)/no_of_training_images*100
				if not image_idx%100:
					sys.stdout.write("\r Loading training data. {}% done".format((float(image_idx)/no_of_training_images)*100))
					sys.stdout.flush()

		sys.stdout.write("\n")
		y_train_one_hot = convert_to_one_hot(y_train, no_of_classes)
		#print "{} : {}".format(y_train[-1], y_train_one_hot[-1])
		#print "{} : {}".format(y_train[0], y_train_one_hot[0])
		#print "{} : {}".format(y_train[100], y_train_one_hot[100])
		return (X_train, y_train_one_hot)

	def convert_to_one_hot(y_normal, no_of_classes):
		y_one_hot = np.zeros((len(y_normal), no_of_classes+1))
		y_one_hot[np.arange(len(y_normal)), y_normal] = 1
		#print y_one_hot
		y_one_hot = np.delete(y_one_hot,0,axis=1) # delete first column coz we dont care about 0 label
		#print y_one_hot
		return y_one_hot

	def test_convert_to_one_hot():
		y_normal = [4,2,1,4,3,1,2,3]
		no_of_classes = 4
		y_one_hot = convert_to_one_hot(y_normal, no_of_classes)
		for idx in range(len(y_normal)):
			print "{} : {}".format(y_normal[idx], y_one_hot[idx])

	def make_three_channel(x):
		x = K.repeat_elements(x,3,axis=2)
		return x

if __name__=='__main__':
	CUBS = CUB_Loader(flag_split_train_test=0)
	CUBS._get_data_statistics(details_per_dir=0)
	CUBS._split_into_train_and_test()
	# CUBS._cubs_into_ram_all()