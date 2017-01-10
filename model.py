from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Input, Flatten
from keras.layers.convolutional import Convolution2D, UpSampling2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import SGD
import cv2, os, sys
import numpy as np
import cubs_loader
from keras import backend as K
from keras.utils import np_utils
from PIL import Image
from random import shuffle
#K.set_image_dim_ordering('th')

# random segnet like hack model
def generator_model():
	# in original gan, recall that relu everywhere except output layer which uses tanh. todo leaky relu
	# batch norm everywhere except generator output and discriminator input
	print('Generator')
	model = Sequential()
	
	# encoder
	model.add(Convolution2D(64, 7, 7, border_mode='same', activation=None))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2), border_mode='same')) #TODO to stride=2 or not?

	model.add(Convolution2D(128,3,3,border_mode='same'))
	# model.add(Convolution2D(128,3,3,border_mode='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2), border_mode='same'))

	model.add(Convolution2D(256, 3, 3, border_mode='same', activation=None))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2), border_mode='same'))

	model.add(Convolution2D(512, 3, 3, border_mode='same', activation=None))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	# decoder
	model.add(Convolution2D(512,3,3,border_mode='same', activation=None))
	model.add(BatchNormalization())

	model.add(Upsampling2D(size=(2,2))) #todo
	model.add(Convolution2D(256,3,3,border_mode='same', activation=None))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(Upsampling2D(size=(2,2))) #todo
	model.add(Convolution2D(128,3,3,border_mode='same', activation=None))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(Upsampling2D(size=(2,2))) #todo
	model.add(Convolution2D(64,3,3,border_mode='same', activation=None))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(Convolution2D(1,1,1, border_mode='same', activation=None)) # single mask output using 1*! convs
	model.add(Activation('relu'))

	# softmax over spatial dimensions as it's a probability mask
	model.add(Reshape(224*224)) # keras applies softmax on trailing dim.  
	model.add(Activation(softmax))
	model.add(Reshape(224,224))
	return model

# vgg19 https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d
def discriminator_model():
	print('Discriminator')
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(200, activation='softmax')) #200 number of birds
	
	return model

# todo implement a masking layer?
def discriminator_on_masked_generator_output(input_tensor, generator, discriminator):
	model = Sequential()
	model.add(generator)
	model.add(Merge([input_tensor, generator], mode='mul')) # mask input image with generator's output
	# todo should this combined model be inside main/train or here. 
	discriminator.trainable = False
	model.add(discriminator)

# def train(no_of_epochs, batch_size):
# 	generator = generator_model()
# 	discriminator = discriminator_model()
# 	combined_model = discriminator_on_masked_generator_output()

# 	# todo check loss. need to add regulalarization to generator
# 	generator.compile(loss='categorical_crossentropy', optimizer='adadelta')
# 	discriminator.compile(loss='categorical_crossentropy', optimizer='adadelta')

# 	model.fit_generator(train_generator, samples_per_epoch=2000, nb_epoch=50)

def check_discriminator_training():
	discriminator = discriminator_model()
	discriminator.compile(loss='categorical_crossentropy', optimizer='adadelta')

	train_datagen = ImageDataGenerator(
					rescale=1./255,
					rotation_range=40,
					width_shift_range=0.2,
					height_shift_range=0.2,
					shear_range=0.2,
					zoom_range=0.2,
					horizontal_flip=True)

	train_generator = train_datagen.flow_from_directory(
						'data/train',
						target_size = (224,224),
						batch_size = 16,
						class_mode = 'categorical')

	discriminator.fit_generator(train_generator, samples_per_epoch=2000, nb_epoch=50)

# load everything into ram and manually generate batches. will take ~10GB for 6000 224*224 images
def cubs_into_ram_all():
	IMAGE_SIZE = (224,224)
	NO_OF_TRAIN_IMAGES = 5994 # find data/train -type f | wc -l
	train_dir = os.path.join(os.getcwd(),'data/train')
	all_sub_dirs=sorted(os.listdir(train_dir))
	NO_OF_CLASSES = len(all_sub_dirs)
	X_train = np.empty((NO_OF_TRAIN_IMAGES,3)+IMAGE_SIZE, dtype='float32') #dtype?
	y_train = np.empty(NO_OF_TRAIN_IMAGES, dtype='int')
	# read, preprocess and dump all in single np array
	category_idx = 0
	image_idx = 0

	print "no_of_classes : {}".format(NO_OF_CLASSES)
	print "no_of_training_images : {}".format(NO_OF_TRAIN_IMAGES)
	for each_dir in all_sub_dirs:
		curr_label = category_idx
		all_images_category = os.listdir(os.path.join(train_dir, each_dir))
		category_idx += 1
		#print category_idx
		for each_image in all_images_category:
			image_file = os.path.join(os.path.join(train_dir, each_dir, each_image))
			#print image_file
			img = Image.open(image_file)
			img = img.convert('RGB') # ensure 3 channel
			img = img.resize(IMAGE_SIZE, resample=Image.NEAREST)
			img_array = np.asarray(img, dtype='float32')
			if len(img_array.shape) == 3:
				img_array = img_array.transpose(2, 0, 1)
			elif len(img_array.shape) == 2:
				img_array = img_array.reshape((1, x.shape[0], x.shape[1]))
			X_train[image_idx, ...] = img_array
			y_train[image_idx] = category_idx #Labels are one indexed
			image_idx += 1
			#print image_idx
			percent_done = float(image_idx)/NO_OF_TRAIN_IMAGES*100
			if not image_idx%100:
				sys.stdout.write("\r Loading training data. {}% done".format((float(image_idx)/NO_OF_TRAIN_IMAGES)*100))
				sys.stdout.flush()

	sys.stdout.write("\n")
	y_train_one_hot = convert_to_one_hot(y_train, NO_OF_CLASSES)
	#print "{} : {}".format(y_train[-1], y_train_one_hot[-1])
        #print "{} : {}".format(y_train[0], y_train_one_hot[0])
        #print "{} : {}".format(y_train[100], y_train_one_hot[100])

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

if __name__=='__main__':
	cubs_into_ram_all()
	#check_discriminator_training()
	#test_convert_to_one_hot()
