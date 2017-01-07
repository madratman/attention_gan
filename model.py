from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Input, Flatten
from keras.layers.convolutional import Convolution2D, UpSampling2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import SGD
import cv2
import numpy as np
import cubs_loader
from keras import backend as K

K.set_image_dim_ordering('th')

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

if __name__=='__main__':
	check_discriminator_training()
