import os, cv2, sys
import numpy as np
from pprint import pprint
#if not (os.uname()[1]=='ratneshmadaan-Inspiron-N5010'):
#	from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image
import random
from keras.utils import np_utils as keras_np_utils
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
		self.avg_no_of_images_per_class = 59
		self.no_of_background_images = self.avg_no_of_images_per_class * 2
		self.no_of_images_in_dataset = 11788
		self.min_height_bgd = 200
		self.min_width_bgd = 200
		self.background_dir = os.path.join(self.image_dir, '201.Background')
		self.background_dir_basename = '201.Background'
		self.background_image_path_list = []
		self.no_of_training_images = 5994 + self.no_of_background_images//2
		self.no_of_testing_images = 5794 + (self.no_of_background_images - self.no_of_background_images//2)
		self.no_of_classes = 200 + 1

		if os.path.exists(self.train_dir):
			self.split_done = True

		# http://stackoverflow.com/questions/1747817/create-a-dictionary-with-list-comprehension-in-python
		# d = dict((key, value) for (key, value) in iterable)
		self.image_id_name_map = {}
		with open(self.image_id_mapping_file) as f:
			self.image_id_name_map = dict( (int(line.split()[0]), line.split()[1]) for line in f.readlines())

		# this is like : 8018: ['104.0', '95.0', '212.0', '154.0']
		self.bboxes_map = {}
		with open(self.bounding_box_file) as f:
			for line in f.readlines():
				image_idx = int(line.split()[0])
				bbox = line.split()[1:]
				bbox = [int(float(x)) for x in bbox]
				self.bboxes_map[image_idx] = bbox

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

			training_image_paths = [self.image_id_name_map[train_idx] for train_idx in train_idx_list]
			testing_image_paths = [self.image_id_name_map[test_idx] for test_idx in test_idx_list]
			# pprint(training_image_paths)
			# pprint(testing_image_paths)
			self._generate_background_images(no_of_images=self.no_of_background_images) # this function fills self.background_image_path_list as well
			#print self.background_image_path_list
			
			training_image_paths.append(self.background_image_path_list[:self.no_of_background_images//2]) #take first half as they are generated randomly anyway
			testing_image_paths.append(self.background_image_path_list[self.no_of_background_images//2:]) #take first half as they are generated randomly anyway
			# now we need to flatten the meta list (as background image names are in a list themselves)
		 	training_image_paths = self.flatten_list(training_image_paths)
			testing_image_paths = self.flatten_list(testing_image_paths)
			#pprint(training_image_paths)
			#pprint(testing_image_paths)

			for idx in range(len(training_image_paths)):
				directory_name = training_image_paths[idx].split('/')[0]
				image_basename = training_image_paths[idx].split('/')[1] 
				
				if not os.path.exists(os.path.join(self.train_dir, directory_name)):
					os.makedirs(os.path.join(self.train_dir, directory_name))

				new_filename = os.path.join(self.train_dir, directory_name, image_basename)
				old_filename = os.path.join(self.image_dir, training_image_paths[idx])
				
				# move file
				#print new_filename
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
		self.avg_no_of_images_per_class = int(no_of_images_arr.mean())

	def _generate_background_images(self, no_of_images=None):
		if not (os.path.exists(self.background_dir)):
			os.makedirs(self.background_dir)
		if no_of_images is None:
			no_of_images = self.no_of_background_images
		no_of_images_done = 0
		no_of_total_trials = 0
		self.background_image_path_list = []
		while no_of_images_done < self.no_of_background_images:
			no_of_total_trials += 1
			rand_idx = random.randint(1, self.no_of_images_in_dataset)
			curr_img = self._remove_patch_from_img(rand_idx)
			if (curr_img.shape[0] > self.min_width_bgd) and (curr_img.shape[1] > self.min_height_bgd):
				no_of_images_done += 1
				filename_base = 'bgd_img_'+str(no_of_images_done).zfill(len(str(no_of_images)))+'.png'
				filename_full = os.path.join(self.background_dir, filename_base)
				cv2.imwrite(filename_full, curr_img)
				self.background_image_path_list.append(self.background_dir_basename + '/' + filename_base)
			else:
				continue
		print "Generate {} images but I tried {} no of times".format(no_of_images, no_of_total_trials)

	def _remove_patch_from_img(self, image_idx, imshow_flag=0):
		image_path = self.image_id_name_map[image_idx]
		bbox = self.bboxes_map[image_idx] # x, y, width, height
		img_array = cv2.imread(os.path.join(self.image_dir, image_path))
		width = bbox[2]
		height = bbox[3]
		if width <= height:
			leftmost = bbox[0]
			left_side_of_bbox = img_array[:, 0:leftmost]
			right_side_of_bbox = img_array[:, leftmost+width:]
			img_minus_bbox = np.concatenate((left_side_of_bbox,right_side_of_bbox), axis=1)
		else:
			topmost = bbox[1]
			above_the_box = img_array[0:topmost, :]
			below_the_box = img_array[topmost+height:, :]
			img_minus_bbox = np.concatenate((above_the_box, below_the_box), axis=0)

		if imshow_flag==1:
			cv2.rectangle(img_array, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0,0,255), 4)
			cv2.imshow("img_minus_bbox", img_minus_bbox)
			cv2.imshow("img_orig", img_array)
			cv2.waitKey(500)

		return img_minus_bbox

	def _test_remove_patch_from_img(self, no_of_images_to_test=100, imshow_flag=1):
		# dear future self, note that image_idx in all the dicts is 1 indexed due to dataset convention
		for n in range(no_of_images_to_test):
			self._remove_patch_from_img(n+1, imshow_flag)

	# load everything into ram and manually generate batches. will take ~10GB for 6000 224*224 images
	def _cubs_into_ram_all(self):
		if self.split_done == False:
			raise ValueError("You haven't split the data!!!")
		image_size = (224,224)
		subdir_list = sorted(os.listdir(self.train_dir))
		X_train = np.empty((self.no_of_training_images,3)+image_size, dtype='float32') #dtype?
		y_train = np.empty(self.no_of_training_images, dtype='int')
		# read, preprocess and dump all in single np array
		image_idx_global = 0

		print "no_of_classes : {}".format(self.no_of_classes)
		print "no_of_training_images : {}".format(self.no_of_training_images)

		for subdir_idx in range(len(subdir_list)):
			list_of_images = os.listdir(os.path.join(self.train_dir, subdir_list[subdir_idx]))
			#print subdir_list[subdir_idx]
			for image_idx in range(len(list_of_images)):
				image_file = os.path.join(os.path.join(self.train_dir, subdir_list[subdir_idx], list_of_images[image_idx]))
				#if subdir_idx == 200:
					#print image_file
					#print subdir_idx, image_idx_global
				img = Image.open(image_file)
				img = img.convert('RGB') # ensure 3 channel
				img = img.resize(image_size, resample=Image.NEAREST)
				img_array = np.asarray(img, dtype='float32')
				if len(img_array.shape) == 3:
					img_array = img_array.transpose(2, 0, 1)
				elif len(img_array.shape) == 2:
					img_array = img_array.reshape((1, x.shape[0], x.shape[1]))
				X_train[image_idx_global, ...] = img_array
				y_train[image_idx_global] = subdir_idx
				image_idx_global += 1
				if not image_idx_global%100:
					sys.stdout.write("\r Loading training data. {}% done".format((float(image_idx_global)/self.no_of_training_images)*100))
					sys.stdout.flush()

		sys.stdout.write("\n Done! \n")

		print "X_train[:,1-2-3,:,:].mean()", X_train[:,0,:,:].mean(), X_train[:,1,:,:].mean(), X_train[:,2,:,:].mean()
		print "standardizing"
		X_train[:,0,:,:] -= X_train[:,0,:,:].mean()
		X_train[:,1,:,:] -= X_train[:,1,:,:].mean()
		X_train[:,2,:,:] -= X_train[:,2,:,:].mean() 
		print "X_train[:,1-2-3,:,:].mean()", X_train[:,0,:,:].mean(), X_train[:,1,:,:].mean(), X_train[:,2,:,:].mean()
		print "X_train[:,1-2-3,:,:].std()", X_train[:,0,:,:].std(), X_train[:,1,:,:].std(), X_train[:,2,:,:].std() 

		#print y_train.shape, y_train.max(), y_train.argmax()
		#print y_train[5993], y_train[5994], y_train[-1]
		y_train_one_hot = keras_np_utils.to_categorical(y_train, nb_classes=self.no_of_classes)
		#y_train_one_hot = self.convert_to_one_hot(y_train, no_of_classes)
		#print "{} : {}".format(y_train[-1], y_train_one_hot[-1])
		#print "{} : {}".format(y_train[0], y_train_one_hot[0])
		#print "{} : {}".format(y_train[100], y_train_one_hot[100])
		return (X_train, y_train_one_hot)

	def convert_to_one_hot(self, y_normal, no_of_classes):
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

	def flatten_list(self, xs):
		result = []
		if isinstance(xs, (list, tuple)):
			for x in xs:
				result.extend(self.flatten_list(x))
		else:
			result.append(xs)
		return result

#if __name__=='__main__':
	#CUBS = CUB_Loader(flag_split_train_test=1)
	# CUBS._get_data_statistics(details_per_dir=0)
	
	#if ((not CUBS.split_done) and (CUBS.flag_split_train_test)):
	#	CUBS._split_into_train_and_test()
	#if CUBS.split_done:
	#	CUBS._cubs_into_ram_all()
	
	#CUBS._cubs_into_ram_all()
	# CUBS._test_remove_patch_from_img(no_of_images_to_test=199)
	#CUBS._generate_background_images(no_of_images=60)
