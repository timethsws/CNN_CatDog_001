# Convolutional Neural Network


# Building the network

# Importing the libraries
from keras.models import Sequential 
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Initialising the CNN
classifier = Sequential()

# Step 1 : Convolution
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation="relu"))

# Step 2 : Max Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Step 3 : Flattening
classifier.add(Flatten())

# Step 4 : Fully Connected layer
classifier.add(Dense(128,activation="relu"))

# Output Layer
classifier.add(Dense(1, activation="sigmoid"))

# Compile the CNN
classifier.compile(optimizer= "adam", loss="binary_crossentropy",metrics=["accuracy"])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        train_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        nb_val_samples=2000)
