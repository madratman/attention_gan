from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Input
from keras.layers.convolutional import Convolution2D, Upsampling2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
import cv2
import numpy as np

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
	model.add(Convolution2D(512,3,3,border_mode='same'))
	model.add(BatchNormalization())

	model.add(Upsampling2D(size=(2,2))) #todo
	model.add(Convolution2D(256,3,3,border_mode='same'))
	model.add(BatchNormalization())

	model.add(Upsampling2D(size=(2,2))) #todo
	model.add(Convolution2D(128,3,3,border_mode='same'))
	model.add(BatchNormalization())

	model.add(Upsampling2D(size=(2,2))) #todo
	model.add(Convolution2D(64,3,3,border_mode='same'))
	model.add(BatchNormalization())

	#todo this might be wrong
	model.add(Convolution2D(1,1,1, border_mode='same')) # single mask output using 1*! convs
	# model.compile(loss="categorical_crossentropy", optimizer='adadelta')

	return model

# vgg19 https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d
def discriminator_model():
	print('Discriminator')
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
	
	# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True) # todo adadelta
	# model.compile(optimizer='adadelta', loss='categorical_crossentropy')

	return model

def train():
	input_to_gan = Input(shape=(batch_size,3,224,224))