import random

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import numpy as np

IMG_HEIGHT = 224
IMG_WIDTH = 224

PATH = os.path.join(r"C:\Users\alec\.keras\datasets\skin-cancer-malignant-vs-benign")
validation_dir = os.path.join(PATH, 'test')

test_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

# IMPORTANT: Set Shuffle to False to preserve the dataset order needed for comparing 
# an actual class against the predicted result
test_data = test_generator.flow_from_directory(batch_size=15,
                                               directory=validation_dir,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
                                               class_mode='binary',
                                               shuffle=False) 

# specify imagenet mean values for centering
#datagen_mean = [123.68, 116.779, 103.939]

# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(IMG_HEIGHT, IMG_WIDTH))
    # convert to array
    img = img_to_array(img)
    img = img.astype('float32')
    img = img/255.0
    # expand dimensions
    img = np.expand_dims(img, axis = 0)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, IMG_HEIGHT, IMG_WIDTH, 3)
    return img
    

# load model
MODEL_PATH = "C:\\Users\\alec\\Desktop\\VSC\\moles\\backup_Oct_19_at_4_16pm\\model_moles_cnn.h5"
#MODEL_PATH = "C:\\Users\\alec\\Desktop\\VSC\\moles\\backup_Oct_19_at_5_05pm\\model_moles_cnn.h5"
loaded_model = tf.keras.models.load_model(MODEL_PATH)
print("Loaded model from disk")

#opt = SGD(lr=0.001, momentum=0.9)
#loaded_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# load a sample image and predict the class
def predict_image(filename, match):
    img = load_image(filename)
    result = loaded_model.predict(img)
    if result[0] < 0.5:
        val = 0
    elif result[0] > 0.5:
        val = 1
    print(result[0])
    if (val) == match:
        print("Correct")
    else:
        print("Incorrect!")

# run analysis

PATH = "C:\\Users\\alec\\.keras\\datasets\\skin-cancer-malignant-vs-benign\\"

 
#predict_image(PATH + "test\\benign\\" + "54.jpg", 0)
#predict_image(PATH + "test\\benign\\" + "548.jpg", 0)
#predict_image(PATH + "test\\benign\\" + "8.jpg", 0)
#predict_image(PATH + "test\\benign\\" + "61.jpg", 0)
#predict_image(PATH + "test\\benign\\" + "1655.jpg", 0)
#predict_image(PATH + "test\\benign\\" + "1711.jpg", 0)
#predict_image(PATH + "test\\benign\\" + "83.jpg", 0)
#predict_image(PATH + "test\\benign\\" + "268.jpg", 0)
#predict_image(PATH + "test\\malignant\\" + "1347.jpg", 1)
#predict_image(PATH + "test\\malignant\\" + "238.jpg", 1)
#predict_image(PATH + "test\\malignant\\" + "1019.jpg", 1)
#predict_image(PATH + "test\\malignant\\" + "247.jpg", 1)
#predict_image(PATH + "test\\malignant\\" + "826.jpg", 1)
#predict_image(PATH + "test\\malignant\\" + "1197.jpg", 1)
#predict_image(PATH + "test\\malignant\\" + "1402.jpg", 1)
#predict_image(PATH + "test\\malignant\\" + "44.jpg", 1)
#predict_image(PATH + "test\\malignant\\" + "778.jpg", 1)


probabilities = loaded_model.predict_generator(generator=test_data)
y_true = test_data.classes
y_pred = probabilities > 0.5

mat = confusion_matrix(y_true, y_pred)
print('Confusion Matrix')
print(confusion_matrix(test_data.classes, y_pred))
print('Classification Report')
target_names = ['Benign', 'Malignant']
print(classification_report(y_true, y_pred, target_names=target_names))

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

plot_confusion_matrix(cm=mat, target_names=target_names)

