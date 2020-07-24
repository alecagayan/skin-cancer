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
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation, concatenate, UpSampling2D, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.optimizers import RMSprop, SGD, Adam

from sklearn.metrics import classification_report, confusion_matrix


PATH = os.path.join(r"C:\Users\alec\.keras\datasets\skin-cancer-malignant-vs-benign-bad")

#_URL2='https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
#inception_weights_file = tf.keras.utils.get_file('inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5', origin=_URL2)

#_URL2='https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
#resnet_weights_file = tf.keras.utils.get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', origin=_URL2)


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
epochs = 50
steps_per_epoch = total_train // batch_size
initial_lr = 0.00001
#initial_lr = 0.0001
print("Initial learning rate: " + str(initial_lr))

# override the above since we have image generator
steps_per_epoch = 20

IMG_HEIGHT = 224
IMG_WIDTH = 224

# Generator for our training data
train_image_generator = ImageDataGenerator(
                    rescale=1./255,
#                    samplewise_center=True,
                    rotation_range=180,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.2,
                    validation_split=0.4)

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

sample_training_images, _ = next(train_data_gen)
augmented_images = [train_data_gen[0][0][0] for i in range(5)]

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(10,10))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

plotImages(augmented_images)

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
    opt = SGD(lr=initial_lr, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# define more complex cnn model
def define_complex_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', input_shape=(IMG_HEIGHT, IMG_HEIGHT, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(64, (3, 3), activation='relu',  kernel_initializer='he_normal', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), activation='relu',  kernel_initializer='he_normal', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(lr=initial_lr)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def define_unet_model():
    inputs = Input(shape=(IMG_HEIGHT, IMG_HEIGHT, 3))
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)
    model.compile(optimizer = Adam(lr = initial_lr), loss = 'binary_crossentropy', metrics = ['accuracy'])
   
    return model


MODEL_LOAD = False
MODEL_FILE = 'model_moles_alce_ham.h5'

# Pick the model we want to use
model = define_complex_cnn_model()
model.summary()

if MODEL_LOAD:
    if os.path.exists(MODEL_FILE):
        model.load_weights(MODEL_FILE)
        print('Loaded weights from ' + MODEL_FILE)


checkpoint = ModelCheckpoint(MODEL_FILE, monitor='val_accuracy', verbose=1, 
                             save_best_only=True, mode='max')

#Learning rate decay with ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.7)

#early stopping if no improvement
early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=8)

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=steps_per_epoch // 5,  # since validation data is 1/5 of training data
    callbacks=[reduce_lr]
)

#if MODEL_SAVE:
#    model.save('model_' + MODEL_WEIGHTS_FILE)  # save model
#    model.save_weights('weights_' + MODEL_WEIGHTS_FILE)  # always save weights after training or during training

_, acc = model.evaluate_generator(test_data_gen)
print('> %.3f' % (acc * 100.0))

probabilities = model.predict_generator(generator=test_data_gen)
y_true = test_data_gen.classes
y_pred = probabilities > 0.5

#print("probabilities")
#print(probabilities)

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
