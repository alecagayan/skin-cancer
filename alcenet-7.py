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
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
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

from alcetools import plot_confusion_matrix, plot_roc_curve
from metrics import balanced_accuracy

print('Keras Version: ', tensorflow.keras.__version__)

input_dir = r'C:\Users\alec\.keras\datasets\skin-cancer-mnist-ham10k-4classes'
output_dir = r'C:\Users\alec\Desktop\VSC\moles\runs'

train_dir = os.path.join(input_dir, 'train')
valid_dir = os.path.join(input_dir, 'test-gen')
test_dir = os.path.join(input_dir, 'test-gen')

time_stamp = time.strftime("%m-%d-%Y %H:%M:%S")
print("Date:", time_stamp)
print("Train data path:", train_dir)
print("Validation data path:", valid_dir)
print("Test data path:", test_dir)

def fileCount(folder):
    "count the number of files in a directory"
    total = 0
    for root, dirs, files in os.walk(folder):
        total += len(files)
        
    return total

total_train = fileCount(train_dir)
total_valid = fileCount(train_dir)
total_test = fileCount(test_dir)

print("Total training images:", total_train)
print("Total validation images:", total_valid)
print("Total testing images:", total_test)

image_size = 224
batch_size = 30
test_batch_size = 1
num_of_epochs = 20
num_of_classes = 4
initial_lr = 0.001
dropout = 0.3
kernel1 = (5,5)
kernel2 = (3,3)

# Used for generation of logs and images
run_id = 'good-ham-4-cnn-3'

print("Batch size:", batch_size)
print("Epochs:", num_of_epochs)
print("Initial LR:", initial_lr)
print("Dropout:", dropout)
print("Kernel 1:", kernel1)
print("Kernel 2:", kernel2)
print("Run ID:", run_id)

train_steps = total_train // batch_size
valid_steps = total_test // batch_size
test_steps = total_test // test_batch_size

# Set this to False to skip training and run predictions from an existing model
perform_fit = True

# Generator for our test data
image_generator = ImageDataGenerator(
    preprocessing_function=tensorflow.keras.applications.mobilenet.preprocess_input)

# Get training data from directory
train_batches = image_generator.flow_from_directory(batch_size=batch_size,
                                                          directory=train_dir,
                                                          target_size=(image_size, image_size))

# Get validation data from directory
valid_batches = image_generator.flow_from_directory(batch_size=batch_size,
                                                         directory=valid_dir,
                                                         target_size=(image_size, image_size))

# Get test data from directory, shuffle=False causes the test dataset to not be shuffled
test_batches = image_generator.flow_from_directory(batch_size=test_batch_size,
                                                        directory=test_dir,
                                                        target_size=(image_size, image_size),
                                                        shuffle=False)

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


# Instantiate model
model = define_cnn_model()    
print(model.summary())

output_folder = os.path.join(output_dir, run_id)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

filepath = os.path.join(output_folder, "alcenet-7-" + run_id + ".h5")
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

model.compile(Adam(lr=initial_lr), loss='categorical_crossentropy', 
#              metrics=[categorical_accuracy, top_2_accuracy, top_3_accuracy])
              metrics=[balanced_accuracy(num_of_classes), top_2_accuracy, top_3_accuracy])

# Get the labels that are associated with each index
print(valid_batches.class_indices)

#class_weights={
#    0: 1.0, # akiec
#    1: 1.0, # bcc
#    2: 1.0, # bkl
#    3: 1.0, # df
#    4: 3.0, # mel
#    5: 1.0, # nv
#    6: 1.0, # vasc
#}

class_weights = class_weight.compute_class_weight(
               'balanced',
                np.unique(train_batches.classes), 
                train_batches.classes)

print('Class weights:', class_weights)

checkpoint = ModelCheckpoint(filepath, monitor='val_balanced_accuracy', verbose=1, 
                             save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_balanced_accuracy', factor=0.1, patience=8, 
                                   verbose=1, mode='max', min_lr=0.000001)

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
    val_loss, val_cat_acc, val_top_2_acc, val_top_3_acc = model.evaluate_generator(test_batches, steps=test_steps)
    print('Best Epoch:')
    print('val_loss:', val_loss)
    print('val_cat_acc:', val_cat_acc)
    print('val_top_2_acc:', val_top_2_acc)
    print('val_top_3_acc:', val_top_3_acc)

    # display the loss and accuracy curves
    acc = history.history['balanced_accuracy']
    val_acc = history.history['val_balanced_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_top2_acc = history.history['top_2_accuracy']
    val_top2_acc = history.history['val_top_2_accuracy']
    train_top3_acc = history.history['top_3_accuracy']
    val_top3_acc = history.history['val_top_3_accuracy']
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

    plt.figure()
    plt.plot(epochs, train_top2_acc, 'bo', label='Training top2 acc')
    plt.plot(epochs, val_top2_acc, 'b', label='Validation top2 acc')
    plt.title('Training and validation top2 accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_folder, run_id + '_top2_accuracy.png'))

    plt.figure()
    plt.plot(epochs, train_top3_acc, 'bo', label='Training top3 acc')
    plt.plot(epochs, val_top3_acc, 'b', label='Validation top3 acc')
    plt.title('Training and validation top3 accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_folder, run_id + '_top3_accuracy.png'))

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

#fpr, tpr, thresholds = roc_curve(y_true, predictions, pos_label=1)
#auc_rf = auc(fpr, tpr)

#print('AUC:', auc_rf)


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