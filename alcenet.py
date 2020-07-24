from numpy.random import seed
seed(101)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

import pandas as pd
import numpy as np

import tensorflow
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import itertools
import shutil
import cv2
import matplotlib.pyplot as plt
from matplotlib.image import imread

input_dir = r'C:\Users\alec\.keras\datasets\isic-2019-7-classes'

train_dir = os.path.join(input_dir, 'train')
valid_dir = os.path.join(input_dir, 'valid')
test_dir = os.path.join(input_dir, 'test')

def fileCount(folder):
    "count the number of files in a directory"
    total = 0
    for root, dirs, files in os.walk(folder):
        total += len(files)
        
    return total

total_train = fileCount(train_dir)
total_valid = fileCount(valid_dir)
total_test = fileCount(test_dir)

print("Total training images:", total_train)
print("Total validation images:", total_valid)
print("Total testing images:", total_test)

image_size = 224
batch_size = 30
test_batch_size = 1
num_of_epochs = 20

train_steps = total_train // batch_size
valid_steps = total_valid // batch_size
test_steps = total_test // test_batch_size

# Load the mask file
mask_file = os.path.join(input_dir, 'mask1.png')
mask_img = imread(mask_file)
#plt.imshow(mask_img)
#plt.show()

# Test that the mask is working
#test_file = os.path.join(input_dir, r'train\bcc\ISIC_0054435.jpg')
#test_img = imread(test_file)
#test_img = cv2.resize(test_img, (image_size, image_size))
#plt.imshow(test_img)
#plt.show()

#idx = (mask_img<1)
#test_img[idx] = mask_img[idx]
#plt.imshow(test_img)
#plt.show()

#kruk
def mask_input(*args, **kwargs):
    "Preprocess images and apply the mask"
    x = tensorflow.keras.applications.mobilenet.preprocess_input(*args, **kwargs)
    idx = (mask_img<1)
    x[idx] = mask_img[idx]
    return x

datagen = ImageDataGenerator(
    preprocessing_function= \
    tensorflow.keras.applications.mobilenet.preprocess_input)

# Generator for our training data
train_image_generator = ImageDataGenerator(
   rescale=1./255)
#    preprocessing_function=tensorflow.keras.applications.mobilenet.preprocess_input)
#    ,
#    rotation_range=180,
#    width_shift_range=.15,
#    height_shift_range=.15,
#    horizontal_flip=True,
#    zoom_range=0.1)

# Generator for our test data
#test_image_generator = ImageDataGenerator(rescale=1./255)
test_image_generator = ImageDataGenerator(
   rescale=1./255)
#    preprocessing_function=tensorflow.keras.applications.mobilenet.preprocess_input)


# Get training data from directory
train_batches = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                          directory=train_dir,
                                                          shuffle=True,
                                                          class_mode="categorical",
                                                          target_size=(image_size, image_size))

# Get validation data from directory
valid_batches = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                         directory=valid_dir,
                                                         shuffle=True,
                                                         class_mode="categorical",
                                                         target_size=(image_size, image_size))

# Get test data from directory, shuffle=False causes the test dataset to not be shuffled
test_batches = test_image_generator.flow_from_directory(batch_size=test_batch_size,
                                                        directory=test_dir,
                                                        target_size=(image_size, image_size),
                                                        shuffle=False)

filepath = ""

def define_mobilenet_model():
    # Load MobileNet
    mobile = tensorflow.keras.applications.mobilenet.MobileNet()
#    print(len(mobile.layers))

    #create model

    # Exclude the last 5 layers of the above model.
    # This will include all layers up to and including global_average_pooling2d_1
    x = mobile.layers[-6].output

    # Create a new dense layer for predictions
    # 7 corresponds to the number of classes
    x = Dropout(0.25)(x)
    predictions = Dense(7, activation='softmax')(x)

    # inputs=mobile.input selects the input layer, outputs=predictions refers to the
    # dense layer we created above.
    model = Model(inputs=mobile.input, outputs=predictions)

#    for layer in model.layers[:-23]:
#        layer.trainable = False

    return model

# define not retarded model
def define_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', 
    #kernel_initializer='he_uniform', 
    padding='same', input_shape=(image_size, image_size, 3)))
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
    model.add(Dense(7, activation='softmax'))
    # compile model
#    opt = SGD(lr=0.001, momentum=0.9)
#    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Instantiate model
model = define_cnn_model()    
print(model.summary())

filepath = "alcenet-cnn.h5"
if os.path.isfile(filepath):
    model.load_weights(filepath)
    print('Loaded weights from: ' + filepath)
else:
    print('Path not found: ' + filepath)

# Define Top2 and Top3 Accuracy

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

model.compile(Adam(lr=0.01), loss='categorical_crossentropy', 
              metrics=[categorical_accuracy, top_2_accuracy, top_3_accuracy])

# Get the labels that are associated with each index
print(valid_batches.class_indices)

class_weights={
    0: 1.0, # akiec
    1: 1.0, # bcc
    2: 1.0, # bkl
    3: 1.0, # df
    4: 3.0, # mel # Try to make the model more sensitive to Melanoma.
    5: 1.0, # nv
    6: 1.0, # vasc
}

#class_weights={
#    0: 0.25, # akiec
#    1: 0.07, # bcc
#    2: 0.1, # bkl
#    3: 1.0, # df
#    4: 0.15, # mel # Try to make the model more sensitive to Melanoma.
#    5: 0.05, # nv
#    6: 1.0, # vasc
#}



checkpoint = ModelCheckpoint(filepath, monitor='val_top_3_accuracy', verbose=1, 
                             save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_top_3_accuracy', factor=0.5, patience=2, 
                                   verbose=1, mode='max', min_lr=0.00001)

#checkpoint = ModelCheckpoint(filepath, monitor='val_categorical_accuracy', verbose=1, 
#                             save_best_only=True, mode='max')

#reduce_lr = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=2, 
#                                   verbose=1, mode='max', min_lr=0.00001)
                              
                              
callbacks_list = [checkpoint, reduce_lr]


history = model.fit_generator(train_batches, steps_per_epoch=train_steps, 
                              class_weight=class_weights,
                    validation_data=valid_batches,
                    validation_steps=valid_steps,
                    epochs=num_of_epochs, verbose=1,
                   callbacks=callbacks_list)

# get the metric names so we can use evaulate_generator
print(model.metrics_names)

# Here the the last epoch will be used.
val_loss, val_cat_acc, val_top_2_acc, val_top_3_acc = \
model.evaluate_generator(test_batches, steps=test_steps)

print('Last Epoch:')
print('val_loss:', val_loss)
print('val_cat_acc:', val_cat_acc)
print('val_top_2_acc:', val_top_2_acc)
print('val_top_3_acc:', val_top_3_acc)


# Here the best epoch will be used.

model.load_weights(filepath)

val_loss, val_cat_acc, val_top_2_acc, val_top_3_acc = \
model.evaluate_generator(test_batches, steps=test_steps)
print('Best Epoch:')
print('val_loss:', val_loss)
print('val_cat_acc:', val_cat_acc)
print('val_top_2_acc:', val_top_2_acc)
print('val_top_3_acc:', val_top_3_acc)


# display the loss and accuracy curves

acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
train_top2_acc = history.history['top_2_accuracy']
val_top2_acc = history.history['val_top_2_accuracy']
train_top3_acc = history.history['top_3_accuracy']
val_top3_acc = history.history['val_top_3_accuracy']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.figure()

plt.plot(epochs, acc, 'bo', label='Training cat acc')
plt.plot(epochs, val_acc, 'b', label='Validation cat acc')
plt.title('Training and validation cat accuracy')
plt.legend()
plt.figure()


plt.plot(epochs, train_top2_acc, 'bo', label='Training top2 acc')
plt.plot(epochs, val_top2_acc, 'b', label='Validation top2 acc')
plt.title('Training and validation top2 accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, train_top3_acc, 'bo', label='Training top3 acc')
plt.plot(epochs, val_top3_acc, 'b', label='Validation top3 acc')
plt.title('Training and validation top3 accuracy')
plt.legend()

plt.show()

test_labels = test_batches.classes

print(test_labels)
print(test_batches.class_indices)

predictions = model.predict_generator(test_batches, steps=test_steps, verbose=1)

print(predictions.shape)

# Source: Scikit Learn website
# http://scikit-learn.org/stable/auto_examples/
# model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-
# selection-plot-confusion-matrix-py


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# argmax returns the index of the max value in a row
cm = confusion_matrix(test_labels, predictions.argmax(axis=1))

# Define the labels of the class indices. These need to match the 
# order shown above.
cm_plot_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel','nv', 'vasc']

plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')

# Get the index of the class with the highest probability score
y_pred = np.argmax(predictions, axis=1)

# Get the labels of the test images.
y_true = test_batches.classes

# Generate a classification report
report = classification_report(y_true, y_pred, target_names=cm_plot_labels)

print(report)