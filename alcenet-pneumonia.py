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
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D, SeparableConv2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import shutil
import cv2
import matplotlib.pyplot as plt
from matplotlib.image import imread
from itertools import cycle
import time
import random

from alcetools import plot_confusion_matrix, plot_roc_curve
from metrics import balanced_accuracy

print('Keras Version: ', tensorflow.keras.__version__)

input_dir = r'C:\Users\alec\Desktop\disease'
output_dir = r'C:\Users\alec\Desktop\VSC\moles\runs'

train_dir = os.path.join(input_dir, 'train')
#valid_dir = os.path.join(input_dir, 'val')
test_dir = os.path.join(input_dir, 'test')

time_stamp = time.strftime("%m-%d-%Y %H:%M:%S")
print("Date:", time_stamp)
print("Train data path:", train_dir)
#print("Validation data path:", valid_dir)
print("Test data path:", test_dir)

def fileCount(folder):
    "count the number of files in a directory"
    total = 0
    for root, dirs, files in os.walk(folder):
        total += len(files)
        
    return total

total_train = fileCount(train_dir)
#total_valid = fileCount(train_dir)
total_test = fileCount(test_dir)

print("Total training images:", total_train)
#print("Total validation images:", total_valid)
print("Total testing images:", total_test)

image_size = 224
batch_size = 30
test_batch_size = 1
num_of_epochs = 20
num_of_classes = 2
initial_lr = 0.001
dropout = 0.3
kernel1 = (3,3)
kernel2 = (3,3)
validation_split = 0.15

# Used for generation of logs and images
run_id = 'alcenet-pneumonia-5'

print("Batch size:", batch_size)
print("Epochs:", num_of_epochs)
print("Initial LR:", initial_lr)
print("Dropout:", dropout)
print("Kernel 1:", kernel1)
print("Kernel 2:", kernel2)
print("Validation split:", validation_split)
print("Run ID:", run_id)

train_steps = total_train // batch_size
valid_steps = total_test // batch_size
test_steps = total_test // test_batch_size

# Set this to False to skip training and run predictions from an existing model
perform_fit = True



def load_and_crop_img(path, grayscale=False, color_mode='rgb', target_size=None,
             interpolation='nearest'):
    """Wraps keras_preprocessing.image.utils.loag_img() and adds cropping.
    Cropping method enumarated in interpolation
    # Arguments
        path: Path to image file.
        color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
            The desired image format.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation and crop methods used to resample and crop the image
            if the target size is different from that of the loaded image.
            Methods are delimited by ":" where first part is interpolation and second is crop
            e.g. "lanczos:random".
            Supported interpolation methods are "nearest", "bilinear", "bicubic", "lanczos",
            "box", "hamming" By default, "nearest" is used.
            Supported crop methods are "none", "center", "random".
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """

    print('moo', path)

    # Decode interpolation string. Allowed Crop methods: none, center, random
    interpolation, crop = interpolation.split(":") if ":" in interpolation else (interpolation, "none")  

    if crop == "none":
        return tensorflow.keras.preprocessing.image.utils.load_img(path, 
                                            grayscale=grayscale, 
                                            color_mode=color_mode, 
                                            target_size=target_size,
                                            interpolation=interpolation)

    # Load original size image using Keras
    img = tensorflow.keras.preprocessing.image.utils.load_img(path, 
                                            grayscale=grayscale, 
                                            color_mode=color_mode, 
                                            target_size=None, 
                                            interpolation=interpolation)

    # Crop fraction of total image
    crop_fraction = 0.75
    target_width = target_size[1]
    target_height = target_size[0]

    if target_size is not None:        
        if img.size != (target_width, target_height):

            if crop not in ["center", "random"]:
                raise ValueError('Invalid crop method {} specified.', crop)

            if interpolation not in tensorflow.keras.preprocessing.image.utils._PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(interpolation,
                        ", ".join(tensorflow.keras.preprocessing.image.utils._PIL_INTERPOLATION_METHODS.keys())))
            
            resample = tensorflow.keras.preprocessing.image.utils._PIL_INTERPOLATION_METHODS[interpolation]

            width, height = img.size

            # Resize keeping aspect ratio
            # result shold be no smaller than the targer size, include crop fraction overhead
            target_size_before_crop = (target_width/crop_fraction, target_height/crop_fraction)
            ratio = max(target_size_before_crop[0] / width, target_size_before_crop[1] / height)
            target_size_before_crop_keep_ratio = int(width * ratio), int(height * ratio)
            img = img.resize(target_size_before_crop_keep_ratio, resample=resample)

            width, height = img.size

            if crop == "center":
                left_corner = int(round(width/2)) - int(round(target_width/2))
                top_corner = int(round(height/2)) - int(round(target_height/2))
                return img.crop((left_corner, top_corner, left_corner + target_width, top_corner + target_height))
            elif crop == "random":
                left_shift = random.randint(0, int((width - target_width)))
                down_shift = random.randint(0, int((height - target_height)))
                return img.crop((left_shift, down_shift, target_width + left_shift, target_height + down_shift))

    return img

# Monkey patch
#tensorflow.keras.preprocessing.image.iterator.load_img = load_and_crop_img

# Generator for our test data
image_generator = ImageDataGenerator(
    rescale=1./255,
#    zoom_range=[0.8,0.8],
#    preprocessing_function=tensorflow.keras.applications.mobilenet.preprocess_input,
    validation_split=validation_split)

# Get training data from directory
train_batches = image_generator.flow_from_directory(batch_size=batch_size,
                                                          directory=train_dir,
                                                          subset='training',
                                                          shuffle=True,
#                                                          interpolation = 'lanczos:center', # <--------- center crop
                                                          target_size=(image_size, image_size))

# Get validation data from directory
valid_batches = image_generator.flow_from_directory(batch_size=batch_size,
                                                         directory=train_dir,
                                                         subset='validation',
                                                         shuffle=True,
#                                                         interpolation = 'lanczos:center', # <--------- center crop
                                                         target_size=(image_size, image_size))

# Get test data from directory, shuffle=False causes the test dataset to not be shuffled
test_batches = image_generator.flow_from_directory(batch_size=test_batch_size,
                                                        directory=test_dir,
#                                                        interpolation = 'lanczos:center', # <--------- center crop
                                                        target_size=(image_size, image_size),
                                                        shuffle=False)

train_steps = train_batches.samples // batch_size
valid_steps = valid_batches.samples // batch_size
test_steps = test_batches.samples // test_batch_size

print ("Training steps:", train_steps) 
print ("Validation steps:", valid_steps) 
print ("Testing steps:", test_steps) 

def define_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, kernel1, activation='relu', padding='same', input_shape=(image_size, image_size, 3)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, kernel2, activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, kernel2, activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(256, kernel2, activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))   

    model.add(Conv2D(512, kernel2, activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))   

    model.add(Dropout(dropout))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(num_of_classes, activation='softmax'))

    return model


def define_mobilenet_model():
    # Load MobileNet
    mobile = tensorflow.keras.applications.mobilenet.MobileNet()
    #print(len(mobile.layers))
    #print(mobile.summary())

    #create model

    # Exclude the last 5 layers of the above model.
    # This will include all layers up to and including global_average_pooling2d_1
    x = mobile.layers[-6].output

    # Create a new dense layer for predictions
    # 7 corresponds to the number of classes
    x = Dropout(0.25)(x)
    predictions = Dense(num_of_classes, activation='softmax')(x)

    # inputs=mobile.input selects the input layer, outputs=predictions refers to the
    # dense layer we created above.
    model = Model(inputs=mobile.input, outputs=predictions)

    # trainable layers (from the end)
    trainable_layers = 23

    # only allow the last 23 layers to be trained
    for layer in model.layers[:-trainable_layers]:
        layer.trainable = False

    return model

def define_deepwise_model():
    input_img = Input(shape=(image_size, image_size, 3), name='ImageInput')
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv1_1')(input_img)
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv1_2')(x)
    x = MaxPooling2D((2,2), name='pool1')(x)
    
    x = SeparableConv2D(128, (3,3), activation='relu', padding='same', name='Conv2_1')(x)
    x = SeparableConv2D(128, (3,3), activation='relu', padding='same', name='Conv2_2')(x)
    x = MaxPooling2D((2,2), name='pool2')(x)
    
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_1')(x)
    x = BatchNormalization(name='bn1')(x)
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_3')(x)
    x = MaxPooling2D((2,2), name='pool3')(x)
    
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_1')(x)
    x = BatchNormalization(name='bn3')(x)
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_2')(x)
    x = BatchNormalization(name='bn4')(x)
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_3')(x)
    x = MaxPooling2D((2,2), name='pool4')(x)
    
    x = Flatten(name='flatten')(x)
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dropout(0.25, name='dropout1')(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dropout(0.25, name='dropout2')(x)
    x = Dense(num_of_classes, activation='softmax', name='fc3')(x)
    
    model = Model(inputs=input_img, outputs=x)
    return model

# Instantiate model
model = define_cnn_model()    
print(model.summary())

output_folder = os.path.join(output_dir, run_id)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

filepath = os.path.join(output_folder, "alcenet-2-" + run_id + ".h5")
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

#optimizer = SGD(lr=initial_lr, momentum=0.9)
optimizer = Adam(lr=initial_lr)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', 
#              metrics=[categorical_accuracy])
              metrics=[balanced_accuracy(num_of_classes)])

# Get the labels that are associated with each index
print(valid_batches.class_indices)

#class_weights={
#    0: 1.0, # benign
#    1: 1.0  # malignant (increase the weight)
#}

class_weights = class_weight.compute_class_weight(
               'balanced',
                np.unique(train_batches.classes), 
                train_batches.classes)

print('Class weights:', class_weights)

checkpoint = ModelCheckpoint(filepath, monitor='val_balanced_accuracy', verbose=1, 
                             save_best_only=True, mode='max')

#reduce_lr = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=2, 
reduce_lr = ReduceLROnPlateau(monitor='val_balanced_accuracy', factor=0.5, patience=3, 
                                   verbose=1, mode='max', min_lr=1e-6)

time_stamp = time.strftime("%Y%m%d-%H%M%S")
log_file = os.path.join(output_folder, "training_log_" + run_id + ".csv")
csv_logger = CSVLogger(filename=log_file, separator=',', append=True)
                              
callbacks_list = [checkpoint, reduce_lr, csv_logger]

if perform_fit:
    history = model.fit_generator(train_batches, steps_per_epoch=train_steps, 
                    class_weight=class_weights,
                    validation_data=valid_batches,
                    validation_steps=valid_steps,
                    epochs=num_of_epochs, verbose=1,
                    callbacks=callbacks_list)

# get the metric names so we can use evaulate_generator
print(model.metrics_names)

# Here the the last epoch will be used.
#val_loss, val_cat_acc, val_top_2_acc, val_top_3_acc = model.evaluate_generator(test_batches, steps=test_steps)
#print('Last Epoch:')
#print('val_loss:', val_loss)
#print('val_cat_acc:', val_cat_acc)
#print('val_top_2_acc:', val_top_2_acc)
#print('val_top_3_acc:', val_top_3_acc)

if perform_fit:
    # Here the best epoch will be used.
    model.load_weights(filepath)
    val_loss, val_cat_acc = model.evaluate_generator(test_batches, steps=test_steps)
    print('Best Epoch:')
    print('val_loss:', val_loss)
    print('val_cat_acc:', val_cat_acc)

    # display the loss and accuracy curves
    acc = history.history['balanced_accuracy']
    val_acc = history.history['val_balanced_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(os.path.join(output_folder, run_id + '_loss.png'))
    
    plt.figure()
    plt.plot(epochs, acc, 'bo', label='Training cat acc')
    plt.plot(epochs, val_acc, 'b', label='Validation cat acc')
    plt.title('Training and validation cat accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_folder, run_id + '_top1_accuracy.png'))


#    plt.show()

test_labels = test_batches.classes

#print('test_labels:', test_labels)
#print('test_batches.class_indices:', test_batches.class_indices)

predictions = model.predict_generator(test_batches, steps=test_steps, verbose=1)

print("Predictions shape:", predictions.shape)

# argmax returns the index of the max value in a row
cm = confusion_matrix(test_labels, predictions.argmax(axis=1))

# Get  the labels of the class indices.
cm_plot_labels = test_batches.class_indices.keys()

plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')
plt.savefig(os.path.join(output_folder, run_id + '_cm.png'))

plot_confusion_matrix(cm, cm_plot_labels, normalize=True,title='Normalized Confusion Matrix')
plt.savefig(os.path.join(output_folder, run_id + '_cm_n.png'))

print('predictions:', predictions)


# Get the index of the class with the highest probability score
y_pred = np.argmax(predictions, axis=1)

# Get the labels of the test images.
y_true = test_batches.classes

#print('y_true:', y_true)
#print('y_true shape:', y_true.shape)
#print('y_pred:', y_pred)
#print('y_pred shape:', y_pred.shape)

# Generate a classification report
report = classification_report(y_true, y_pred, target_names=cm_plot_labels)

print(report)

fpr, tpr, thresholds = roc_curve(y_true, y_pred)
auc_rf = auc(fpr, tpr)

print('AUC:', auc_rf)


preds = pd.DataFrame(predictions, columns = test_batches.class_indices.keys())
preds['filename'] = test_batches.filenames
preds['truth'] = preds['filename'].apply(os.path.dirname)
preds['predicted_class'] = preds[list(test_batches.class_indices.keys())].idxmax(1)
print(preds.head())

print(str(np.mean(preds['predicted_class'] == preds['truth']) * 100) + "% Accuracy")

def get_truths(df, class_label):
    y_truth = df['truth'] == class_label
    return y_truth.astype(int).values, df[class_label].values

from scipy import interp

# Compute ROC curve and ROC area for each class
n_classes = len(test_batches.class_indices)
classes = test_batches.class_indices.keys()
lw=2
fpr = dict()
tpr = dict()
roc_auc = dict()
for k,i in test_batches.class_indices.items():
    t, p = get_truths(preds, k)
    fpr[i], tpr[i], _ = roc_curve(t, p)
    roc_auc[i] = auc(fpr[i], tpr[i])
    
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at these points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# average it and compute AUC
mean_tpr /= n_classes

fpr["overall"] = all_fpr
tpr["overall"] = mean_tpr
roc_auc["overall"] = auc(fpr["overall"], tpr["overall"])


print_auc = (lambda x,v: print('{v} AUC: {x:.6f}'.format(v=v, x=x)))
for k,v in test_batches.class_indices.items():
    print_auc(roc_auc[v], k)
print_auc(roc_auc['overall'], "Overall")


# Plot all ROC curves
plt.figure()
#plt.plot(fpr["micro"], tpr["micro"],
#        label='micro-average ROC curve (area = {0:0.2f})'
#            ''.format(roc_auc["micro"]),
#        color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["overall"], tpr["overall"],
        label='Average ROC curve (area = {0:0.2f})'
            ''.format(roc_auc["overall"]),
        color='navy', linestyle=':', linewidth=4)

colors = cycle(['red', 'orange', 'blue', 'purple', 'magenta', 'green', 'grey', 'skyblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
            label='ROC curve of class {0} (area = {1:0.2f})'
            ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig(os.path.join(output_folder, run_id + '_roc.png'))


# Zoom in view of the upper left corner.
plt.figure()
plt.plot(fpr["overall"], tpr["overall"],
        label='Average ROC curve (area = {0:0.2f})'
            ''.format(roc_auc["overall"]),
        color='navy', linestyle=':', linewidth=4)

colors = cycle(['red', 'orange', 'blue', 'purple', 'magenta', 'green', 'grey', 'skyblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
            label='ROC curve of class {0} (area = {1:0.2f})'
            ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 0.4])
plt.ylim([0.6, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig(os.path.join(output_folder, run_id + '_roc2.png'))