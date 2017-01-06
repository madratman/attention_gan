import os, cv2
import numpy as np
from pprint import pprint
# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

FLAG_TRAIN_TEST_SPLIT_FROM_SCRATCH = 1

images_dir = 'data/CUB_200_2011/images'
segmentation_label_dir = 'data/segmentations'
image_id_mapping_file = 'data/CUB_200_2011/images.txt' # this file provides idx for each image. they are NOT sorted alphabetically 
train_test_flag_file = 'data/CUB_200_2011/train_test_split.txt' # this tells if each image specified by the index is in training set(1) or test set(0)
image_class_labels_file = 'data/CUB_200_2011/image_class_labels_file.txt'

# the original cubs dataset has each type of bird's image's in a diff subfolder. Let's first put these into a single train and test folder
# (the split is specified in train_test_split.txt), so that we can use keras' data preprocessors

# meta list has each element in form [idx, flag]. idx goes till 11788, flag=1 it's in training set. 

if FLAG_TRAIN_TEST_SPLIT_FROM_SCRATCH==1:
	meta_list = []
	train_idx_list = []
	test_idx_list = []

	with open(train_test_flag_file) as file:
		for each_line in file:
			curr_list = [int(each_number.strip()) for each_number in each_line.split(' ')]
			# print curr_list
			if curr_list[1]==1:
				train_idx_list.append(curr_list[0])
			else:
				test_idx_list.append(curr_list[0]) 
			meta_list.append(curr_list)

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
		old_filename = os.path.join(images_dir, training_image_paths[idx])
		
		# move file
		os.rename(old_filename, new_filename)

	for idx in range(len(testing_image_paths)):
		directory_name = testing_image_paths[idx].split('/')[0]
		image_basename = testing_image_paths[idx].split('/')[1] 
		
		if not os.path.exists('data/test/'+directory_name):
			os.makedirs('data/test/'+directory_name)

		new_filename = os.path.join('data/test/'+directory_name, image_basename)
		old_filename = os.path.join(images_dir, testing_image_paths[idx])
		
		# move file
		os.rename(old_filename, new_filename)

# keras' image preprocessing datagenerator
normalization?
train_datagen = ImageDataGenerator(
        rescale=1./255,
		rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(
		rescale=1./255)

# visualize 
# img = load_img('data/train/001_Black_Footed_Albatross_0007_796138.jpg')
# x = img_to_array(img)
# print x.dtype
# x = x.reshape((1,)+x.shape)
# y=np.array([1])

# if not os.path.exists('data/preview'):
#         os.makedirs('data/preview')

# i=0
# for batch in datagen.flow(x, y,batch_size=1):
#         batch[0].reshape(batch[0].shape[1:])
#         curr = batch[0]
#         curr_sk = img_as_uint(curr)
#         curr  = curr.astype(int)
#         cv2.imwrite('data/preview/'+str(i)+'.jpeg', curr)
#         io.save('data/preview/'+str(i)+'.png', curr_sk)
#         i+=1
#         print i
#         if i>20:
#                 break

train_generator = train_datagen.flow_from_directory(
					'data/train',
					target_size = (224,224),
					batch_size = 64,
					class_mode = '')

test_generator = test_datagen.flow_from_directory(
					'data/test',
					target_size = (224, 224),
					batch_size = 64,
					class_mode = 'binary')
