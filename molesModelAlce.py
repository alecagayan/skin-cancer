from __future__ import absolute_import, division, print_function, unicode_literals

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.optimizers import RMSprop, SGD

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

PATH = os.path.join(r"C:\Users\alec\.keras\datasets\skin-cancer-malignant-vs-benign-ham-gen")

_URL2='https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
inception_weights_file = tf.keras.utils.get_file('inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5', origin=_URL2)

_URL2='https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
resnet_weights_file = tf.keras.utils.get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', origin=_URL2)


train_dir = os.path.join(PATH, 'train')
test_dir = os.path.join(PATH, 'test')

train_benign_dir = os.path.join(train_dir, 'benign')  # directory with our training benign pictures
train_malignant_dir = os.path.join(train_dir, 'malignant')  # directory with our training malignant pictures
test_benign_dir = os.path.join(test_dir, 'benign')  # directory with our test benign pictures
test_malignant_dir = os.path.join(test_dir, 'malignant')  # directory with our test malignant pictures

num_benign_tr = len(os.listdir(train_benign_dir))
num_malignant_tr = len(os.listdir(train_malignant_dir))

num_benign_test = len(os.listdir(test_benign_dir))
num_malignant_test = len(os.listdir(test_malignant_dir))

total_train = num_benign_tr + num_malignant_tr
total_test = num_benign_test + num_malignant_test

print('Total training benign images:', num_benign_tr)
print('Total training malignant images:', num_malignant_tr)
print('Total test benign images:', num_benign_test)
print('Total test malignant images:', num_malignant_test)
print("--")
print("Total training images:", total_train)
print("Total testing images:", total_test)

batch_size = 15
epochs = 10
steps_per_epoch = total_train // batch_size

# override the above since we have image generator
steps_per_epoch = 500

IMG_HEIGHT = 224
IMG_WIDTH = 224

# Generator for our training data
train_image_generator = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=180,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.2,
                    validation_split=0.2)

# Generator for our test data
test_image_generator = ImageDataGenerator(rescale=1./255)

# Get training data from directory
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           subset='training',
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

# Get validation data from directory
val_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           subset='validation',
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

# Get test data from directory
test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=test_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary',
                                                              shuffle=False)

#sample_training_images, _ = next(train_data_gen)
#augmented_images = [train_data_gen[0][0][0] for i in range(5)]

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

#plotImages(augmented_images)

# define more complex cnn model
def define_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', 
    #kernel_initializer='he_uniform', 
    padding='same', input_shape=(IMG_HEIGHT, IMG_HEIGHT, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', 
    #kernel_initializer='he_uniform', 
    padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', 
    #kernel_initializer='he_uniform', 
    padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# define simple cnn model
def define_simple_cnn_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_HEIGHT, IMG_HEIGHT, 3)))
	model.add(MaxPooling2D((2, 2)))
	#model.add(Dropout(0.2))
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	#model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	#model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	return model

MODEL_LOAD = True
MODEL_FILE = 'model_moles_alce_cnn.h5'

# Pick the model we want to use
model = define_cnn_model()

if MODEL_LOAD:
    if os.path.exists(MODEL_FILE):
        model.load_weights(MODEL_FILE)
        print('Loaded weights from ' + MODEL_FILE)


checkpoint = ModelCheckpoint(MODEL_FILE, monitor='val_accuracy', verbose=1, 
                             save_best_only=True, mode='max')

#Learning rate decay with ReduceLROnPlateau
reduce_lr= ReduceLROnPlateau(monitor='val_accuracy',patience=3,verbose=1,factor=0.7)

# Calculate the weights for each class so that we can balance the data
weights = class_weight.compute_class_weight('balanced',
                                            np.unique(train_data_gen.classes),
                                            train_data_gen.classes)

print('Calculated class weights as', weights)

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=steps_per_epoch // 5,  # since validation data is 1/5 of training data
    class_weight=weights,
    callbacks=[reduce_lr]
)

#if MODEL_SAVE:
#    model.save('model_' + MODEL_WEIGHTS_FILE)  # save model
#    model.save_weights('weights_' + MODEL_WEIGHTS_FILE)  # always save weights after training or during training

model.summary()

_, acc = model.evaluate_generator(test_data_gen)
print('> %.3f' % (acc * 100.0))

probabilities = model.predict_generator(generator=test_data_gen)
y_true = test_data_gen.classes
y_pred = probabilities > 0.5

mat = confusion_matrix(y_true, y_pred)
print('Confusion Matrix')
print(confusion_matrix(test_data_gen.classes, y_pred))
print('Classification Report')
target_names = ['Benign', 'Malignant']
print(classification_report(y_true, y_pred, target_names=target_names))


def summarize_diagnostics(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

summarize_diagnostics(history)
