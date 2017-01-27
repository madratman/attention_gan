from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Input, Flatten, Reshape, merge, Merge, RepeatVector, Lambda # small m is functional api, caps is sequential
from keras.layers.convolutional import Convolution2D, UpSampling2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import SGD
import cv2, os, sys
import numpy as np
import cubs_loader
from keras import backend as K
from keras.utils import np_utils
from random import shuffle
#K.set_image_dim_ordering('th')

# random segnet like hack model
def generator_model():
	# in original gan, recall that relu everywhere except output layer which uses tanh. todo leaky relu
	# batch norm everywhere except generator output and discriminator input
	print('Compiling Generator')
	model = Sequential()
	
	# encoder
	model.add(Convolution2D(64, 7, 7, border_mode='same',input_shape=(3,224,224)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2), border_mode='same')) #TODO to stride=2 or not?

	model.add(Convolution2D(128,3,3,border_mode='same'))
	# model.add(Convolution2D(128,3,3,border_mode='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2), border_mode='same'))

	model.add(Convolution2D(256, 3, 3, border_mode='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2), border_mode='same'))

	model.add(Convolution2D(512, 3, 3, border_mode='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	# decoder
	model.add(Convolution2D(512,3,3,border_mode='same'))
	model.add(BatchNormalization())

	model.add(UpSampling2D(size=(2,2))) #todo
	model.add(Convolution2D(256,3,3,border_mode='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(UpSampling2D(size=(2,2))) #todo
	model.add(Convolution2D(128,3,3,border_mode='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(UpSampling2D(size=(2,2))) #todo
	model.add(Convolution2D(64,3,3,border_mode='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(Convolution2D(1,1,1, border_mode='same')) # single mask output using 1*! convs
	model.add(Activation('relu'))

	# softmax over spatial dimensions as it's a probability mask
	model.add(Reshape((1,224*224))) # keras applies softmax on trailing dim.  
	model.add(Activation('softmax'))
	model.add(Reshape((224,224)))
	return model

# vgg19 https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d
def discriminator_model():
	print('Compiling Discriminator')
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

if __name__=='__main__':
	#check_discriminator_training()
	#test_convert_to_one_hot()

	# todo make config file
	batch_size = 16
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
	generated_mask = Flatten()(generated_mask)
	generated_mask = RepeatVector(3)(generated_mask)
	generated_mask = Reshape((3,224,224))(generated_mask)
	#generated_mask_3_channel = Lambda(make_three_channel)(generated_mask)
	masked_input = merge([input_tensor, generated_mask], mode='mul') # mask input image with generator's output
	discriminator.trainable = False # for the combined model, we keep the discriminator frozen.
	discriminator_output = discriminator(masked_input)
	combined_model = Model(input=input_tensor, output=discriminator_output)
	masked_layer_model = Model(input=input_tensor, output=masked_input)
	# compile models. todo fix losses
	generator.compile(loss='categorical_crossentropy', optimizer="adadelta")
	combined_model.compile(loss='categorical_crossentropy', optimizer="adadelta")
	discriminator.trainable = True
	discriminator.compile(loss='categorical_crossentropy', optimizer="adadelta")

	# this function allows as to get the masked_image from the combined model which we can use to train the discriminator. 
	#get_masked_images = K.function(combined_model.inputs, masked_input.outputs(train=False))

	for epoch_idx in range(no_of_epochs):
		for batch_idx in range(no_of_batches):
			image_batch = X_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
			label_batch = y_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
	
			generated_masks = generator.predict(image_batch)
			masked_images = masked_layer_model.predict(image_batch)
			#masked_images = get_masked_images(image_batch)

			X = np.concatenate((image_batch, masked_images))
			y = np.concatenate((label_batch, label_batch))
			# stack labels as we're going to train masked and non masked images both. todo shuffle or not?

			discriminator_loss = discriminator.train_on_batch(X, y)
			print("Epoch : {0}, Batch : {1} of {2}, discriminator_loss : {3}".format(epoch_idx, batch_idx, no_of_batches, discriminator_loss))
			
			# freeze discriminator while training combined model (or basically the generator)
			discriminator.trainable = False

			#generator_loss = combined_model.train_on_batch(image_batch, generator_labels) # todo what should be generator labels?
			generator_loss=0
			print("Epoch : {0}, Batch : {1} of {2}, generator_loss : {3}".format(epoch_idx, batch_idx, no_of_batches, generator_loss))
			
			discriminator.trainable = True

			# todo save weights, masks, images, etc
