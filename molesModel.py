from __future__ import absolute_import, division, print_function, unicode_literals

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout


from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential

from sklearn.metrics import classification_report, confusion_matrix

import numpy as np
import matplotlib.pyplot as plt

PATH = os.path.join(r"C:\Users\alec\.keras\datasets\skin-cancer-malignant-vs-benign")

_URL2='https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
inception_weights_file = tf.keras.utils.get_file('inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5', origin=_URL2)

_URL2='https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
resnet_weights_file = tf.keras.utils.get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', origin=_URL2)


train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'test')

train_benign_dir = os.path.join(train_dir, 'benign')  # directory with our training benign pictures
train_malignant_dir = os.path.join(train_dir, 'malignant')  # directory with our training malignant pictures
validation_benign_dir = os.path.join(validation_dir, 'benign')  # directory with our validation benign pictures
validation_malignant_dir = os.path.join(validation_dir, 'malignant')  # directory with our validation malignant pictures

num_benign_tr = len(os.listdir(train_benign_dir))
num_malignant_tr = len(os.listdir(train_malignant_dir))

num_benign_val = len(os.listdir(validation_benign_dir))
num_malignant_val = len(os.listdir(validation_malignant_dir))

total_train = num_benign_tr + num_malignant_tr
total_val = num_benign_val + num_malignant_val

print('total training benign images:', num_benign_tr)
print('total training malignant images:', num_malignant_tr)

print('total validation benign images:', num_benign_val)
print('total validation malignant images:', num_malignant_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

batch_size = 15
epochs = 10
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Generator for our training data
train_image_generator = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=180,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.2)

# Generator for our validation data
validation_image_generator = ImageDataGenerator(rescale=1./255)

# Get training data from directory
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

# Get validation data from directory
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
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

# define pre-trained model
def define_inception3_model():
    pre_trained_model = InceptionV3(input_shape=(IMG_HEIGHT, IMG_HEIGHT, 3), include_top=False, weights=None)
    pre_trained_model.load_weights(inception_weights_file)

    for layer in pre_trained_model.layers:
        layer.trainable = False

    last_layer = pre_trained_model.get_layer('mixed7')
    print('last layer output shape:', last_layer.output_shape)
    last_output = last_layer.output

    # Flatten the output layer to 1 dimension
    x = layers.Flatten()(last_output)
    # Add a fully connected layer with 1,024 hidden units and ReLU activation
    x = layers.Dense(1024, activation='relu')(x)
    # Add a dropout rate of 0.2
    x = layers.Dropout(0.2)(x)
    # Add a final sigmoid layer for classification
    x = layers.Dense(1, activation='sigmoid')(x)

    # Configure and compile the model
    model = Model(pre_trained_model.input, x)
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.0001),
                  metrics=['accuracy'])
    return model

# define pre-trained model
def define_resnet50_model():
    model = Sequential()
    model.add(ResNet50(input_shape=(IMG_HEIGHT, IMG_HEIGHT, 3), include_top=False, weights=resnet_weights_file, input_tensor=None, classes=2, pooling='avg'))
#    pre_trained_model.load_weights(resnet_weights_file)

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))

    model.layers[0].trainable = False

    # Compile the model
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy'])
    return model

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
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	#model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
	return model

# define VGG16 model
def define_vgg16_model():
	model = VGG16(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT, IMG_HEIGHT, 3))
	# mark loaded layers as not trainable
	for layer in model.layers:
		layer.trainable = False
	# add new classifier layers
	flat1 = Flatten()(model.layers[-1].output)
	class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
	output = Dense(1, activation='sigmoid')(class1)
	# define new model
	model = Model(inputs=model.inputs, outputs=output)
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model

MODEL_SAVE_LOAD = False
MODEL_WEIGHTS_FILE = 'moles_pre_trained_cnn.h5'

# Pick the model we want to use
model = define_cnn_model()

if MODEL_SAVE_LOAD:
    if os.path.exists('weights_' + MODEL_WEIGHTS_FILE):
        model.load_weights('weights_' + MODEL_WEIGHTS_FILE)
        print('Loaded weights from weights_' + MODEL_WEIGHTS_FILE)

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

if MODEL_SAVE_LOAD:
    model.save('model_' + MODEL_WEIGHTS_FILE)  # save model
    model.save_weights('weights_' + MODEL_WEIGHTS_FILE)  # always save weights after training or during training

model.summary()

_, acc = model.evaluate_generator(val_data_gen, steps=total_val // batch_size)
print('> %.3f' % (acc * 100.0))

probabilities = model.predict_generator(generator=val_data_gen)
y_true = val_data_gen.classes
y_pred = probabilities > 0.5

mat = confusion_matrix(y_true, y_pred)
print('Confusion Matrix')
print(confusion_matrix(val_data_gen.classes, y_pred))
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
