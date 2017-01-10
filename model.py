from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Input, Flatten, merge, Merge # small m is functional api, caps is sequential
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
	image_size = (224,224)
	no_of_training_images = 5994 # find data/train -type f | wc -l
	train_dir = os.path.join(os.getcwd(),'data/train')
	all_sub_dirs=sorted(os.listdir(train_dir))
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
		all_images_category = os.listdir(os.path.join(train_dir, each_dir))
		category_idx += 1
		#print category_idx
		for each_image in all_images_category:
			image_file = os.path.join(os.path.join(train_dir, each_dir, each_image))
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

if __name__=='__main__':
	#check_discriminator_training()
	#test_convert_to_one_hot()

	# todo make config file
	batch_size = 64
	no_of_epochs = 10
	(X_train, y_train) = cubs_into_ram_all()

	# shuffle data
	assert len(X_train) == len(y_train)
	shuff_ind = np.random.permutation(len(y_train))
	X_train = X_train[shuff_ind]
	y_train = y_train[shuff_ind]

	# todo normalize, preprocess data
	no_of_batches = int(X_train.shape[0]/batch_size) # batches per epoch

	discriminator = discriminator_model()
	generator = generator_model()

	# combined model. 
	# todo read the first example https://keras.io/getting-started/functional-api-guide/. We might be able to condition the mask
	# on an embedding of text : caption or question

	input_tensor = Input(shape=(3,224,224), dtype='float32', name='input_tensor') # tocheck shape 
	generated_mask = generator(input_tensor) # all models are callable 
	masked_input = merge([input_tensor, generated_mask], mode='mul') # mask input image with generator's output
	discriminator.trainable = False # for the combined model, we keep the discriminator frozen.
	discriminator_output = discriminator(masked_input)
	combined_model = Model(input=input_tensor, output=discriminator_output)

	# compile models. todo fix losses
	generator.compile(loss='categorical_crossentropy', optimizer="adadelta")
	combined_model.compile(loss='categorical_crossentropy', optimizer="adadelta")
	discriminator.trainable = True
	discriminator.compile(loss='categorical_crossentropy', optimizer="adadelta")

	# this function allows as to get the masked_image from the combined model which we can use to train the discriminator. 
	get_masked_images = theano.function([combined_model.get_input(train=False)], masked_input.get_output(train=False))

	for epoch_idx in range(no_of_epochs):
		for batch_idx in range(no_of_batches):
			image_batch = X_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
			label_batch = y_train[batch_idx*batch_size:(batch_idx+1)*batch_size]

			generated_masks = generator.predict(image_batch)
			masked_images = get_masked_images(image_batch)

			X = np.concatenate((image_batch, masked_images))
			y = np.concatenate((label_batch, label_batch))
			# stack labels as we're going to train masked and non masked images both. todo shuffle or not?

			discriminator_loss = discriminator.train_on_batch(X, y)
			print("Epoch : {0}, Batch : {1} of {2}, discriminator_loss : {3}".format(epoch_idx, batch_idx, no_of_batches, discriminator_loss)
			
			# freeze discriminator while training combined model (or basically the generator)
			discriminator.trainable = False

			generator_loss = combined_model.train_on_batch(image_batch, generator_labels) # todo what should be generator labels?
			print("Epoch : {0}, Batch : {1} of {2}, generator_loss : {3}".format(epoch_idx, batch_idx, no_of_batches, generator_loss)
			
			discriminator.trainable = True

			# todo save weights, masks, images, etc