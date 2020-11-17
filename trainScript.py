# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 22:50:03 2020

@author: rasto
"""
import os
import cv2
import time
import itertools
import imutils
import numpy as np
from PIL import Image
from tqdm import tqdm
import splitfolders as splitter
import matplotlib.pyplot as plt


import tensorflow as tf

from keras.optimizers import Adam, RMSprop
#from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Activation, Flatten, Dense
from keras import backend as K

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

###split the binary data into train/test/valid folers by ratio, only when needed
#splitter.ratio('/home/rastislav/Desktop/bp/brain_tumor_vgg/original/', output="/home/rastislav/Desktop/bp/brain_tumor_vgg/dataset/", seed=1337, ratio=(.7, 0.2,0.1))


#TRAIN_DIR='/home/rastislav/Desktop/bp/brain_tumor_vgg/cropped/train/'
#TEST_DIR='/home/rastislav/Desktop/bp/brain_tumor_vgg/cropped/test/'
#VAL_DIR='/home/rastislav/Desktop/bp/brain_tumor_vgg/cropped/val/'

#TRAIN_DIR='/home/rastislav/Desktop/bp/brain_tumor_vgg/dataset/train/'
#TEST_DIR='/home/rastislav/Desktop/bp/brain_tumor_vgg/dataset/test/'
#VAL_DIR='/home/rastislav/Desktop/bp/brain_tumor_vgg/dataset/val/'

#TRAIN_DIR='/home/rastislav/Desktop/bp/brain_tumor_vgg/dataset-bigger-spl/train/'
#TEST_DIR='/home/rastislav/Desktop/bp/brain_tumor_vgg/dataset-bigger-spl/test/'
#VAL_DIR='/home/rastislav/Desktop/bp/brain_tumor_vgg/dataset-bigger-spl/val/'

TRAIN_DIR='/home/rastislav/Desktop/bp/brain_tumor_vgg/cropped-bg/train/'
TEST_DIR='/home/rastislav/Desktop/bp/brain_tumor_vgg/cropped-bg/test/'
VAL_DIR='/home/rastislav/Desktop/bp/brain_tumor_vgg/cropped-bg/val/'
IMG_SIZE=(128,128)
EPOCHS=20
RANDOM_SEED=123
batch_size = 30
NUM_CLASSES=1


"""
   Expect directory -> dataset path
    for each class(folder) load into np Array 
    @Returns: np_array_of_images, image_class, dictionary_of_classes
"""
def load_data(dir_path,img_size=(100,100)):
    X = []
    y =[]
    labels = dict() # names of classes (folders)
    i=0
    for folder_name in tqdm(sorted(os.listdir(dir_path))):
        if not folder_name.startswith('.'):
            labels[i]=folder_name
            for file in os.listdir(dir_path+folder_name):
                if not file.startswith('.'):
                    img = cv2.imread(dir_path + folder_name + '/' + file)
                    img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_CUBIC )
                    X.append(img)
                    y.append(i)
            i+=1
    X=np.array(X)
    y=np.array(y)
    return X,y,labels

# Get all the data -> train,test,valid and show histogram of them
X_train,y_train,labels = load_data(TRAIN_DIR,IMG_SIZE)
X_test,y_test, _ = load_data(TEST_DIR,IMG_SIZE)
X_val,y_val, _ = load_data(VAL_DIR,IMG_SIZE)

"""
    Show image previews for each class in dataset
        n: examples to print
"""
def print_data_examples(x,y,dictLabels,n):
    for idx in range(len(dictLabels)):
        perClassImages = x[np.argwhere(y==idx)][:n]
        plt.close('all')
        j=10
        i=int(n/j)
        my_dpi=200
        plt.figure(figsize=(15,6),dpi=my_dpi)
        c=1
        for img in perClassImages:
            plt.subplot(i,j,c)
            
            plt.imshow(img[0],interpolation='nearest') 
            c+=1
            plt.xticks([])
            plt.yticks([])
            plt.suptitle('Tumor: ' + dictLabels[idx])
        plt.show()
        
        
print_data_examples(X_train,y_train,labels,10) ## preview unchanged images

def save_preprocessed_images(x_set,y_values,folder_name):
    i=0
    for (img,img_class) in zip(x_set,y_values):
        if (img_class == 0):
            cv2.imwrite(folder_name + "no/" + str(i) + ".jpg", img)
        else:
            cv2.imwrite(folder_name + "yes/" + str(i) + ".jpg", img)
        i+=1

"""
    Having set of images, apply cropping based on extreme points
    Saves the images into folder specified by folder_name
        Returns new set of same, but cropped images
"""
def crop_images(set_imgs, y_values, folder_name, add_pixels_value=0):
    cropped_set = []
    i=0
    for image in set_imgs:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1] # treshold the image
        thresh = cv2.erode(thresh, None, iterations=2) # add series of erosion
        thresh = cv2.dilate(thresh, None, iterations=2) # add dilatitions to remove small regions of noise
        
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        
        # find the extreme points 
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
               
        ADD_PIXELS = add_pixels_value
        new_img = image[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        cropped_set.append(new_img)
        if (y_values[i] == 0):
            cv2.imwrite(folder_name + "no/" + str(i) + ".jpg", new_img)
        else:
            cv2.imwrite(folder_name + "yes/" + str(i) + ".jpg", new_img)
        i+=1
        
    return np.array(cropped_set)



X_train = crop_images(X_train,y_train,"cropped-bg/train/")
X_test = crop_images(X_test, y_test, "cropped-bg/test/")
X_val = crop_images(X_val, y_val, "cropped-bg/val/")


"""
    Get bar chart for organization of classes per train/test/valid
    2 Classes only: no & yes
"""
def imageClassesOrganization():
    N = 3
    no_means = (np.count_nonzero(y_train == 0),np.count_nonzero(y_test == 0),np.count_nonzero(y_val == 0))
    yes_means = (np.count_nonzero(y_train == 1),np.count_nonzero(y_test == 1),np.count_nonzero(y_val == 1))

    ind = np.arange(N)
    width = 0.35
    plt.bar(ind, no_means, width, label='has not tumor')
    plt.bar(ind + width, yes_means, width,label='Has tumor')

    plt.ylabel('Number of images')
    plt.title('Number of images per class')

    plt.xticks(ind + width / 2, ('Training', 'Test', 'Validation',))
    plt.legend(loc='best')
    plt.show()

imageClassesOrganization()



# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.2,
        brightness_range=[0.1, 1.5],
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=preprocess_input
        )


# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
    )

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
    )


# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,  # this is the target directory
        target_size=IMG_SIZE,  # all images will be resized 
        batch_size=batch_size,
        class_mode='binary',  # since we use binary_crossentropy loss, we need binary labels
        seed=RANDOM_SEED,
        )

# this is a similar generator, for validation data
validation_generator = validation_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=12,
        class_mode='binary',
        seed=RANDOM_SEED,
        )

test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        class_mode='binary',
        seed=RANDOM_SEED
        )

# Add also early stopping
es = EarlyStopping(
    monitor='val_accuracy', 
    mode='max',
    patience=12,
)

######################################## Creating model ##########################

if K.image_data_format() == 'channels_first':
    input_shape = (3, IMG_SIZE[0], IMG_SIZE[1])
else:
    input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)

vgg16_weight_path = '../../../.keras/models//vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
base_model = VGG16(
     weights=vgg16_weight_path,
  #  weights="imagenet",
    include_top=False, 
    input_shape=IMG_SIZE + (3,)
    )

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='sigmoid'))

model.layers[0].trainable = False


model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(lr=1e-4),
    metrics=['accuracy']
    )


model.summary()

start = time.time()

history = model.fit_generator(
        train_generator,
        steps_per_epoch=50,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=25,
        callbacks=[es]
        )

model.save_weights('first_try.h5')  # always save your weights after training or during training

end = time.time()
print("Time needed for training: ", end - start, "s.")

def preprocess_imgs(set_name, img_size):
    set_new = []
    for img in set_name:
        img = cv2.resize(
            img,
            dsize=img_size,
            interpolation=cv2.INTER_CUBIC
        )
        set_new.append(preprocess_input(img))
    return np.array(set_new)


X_test_prep = preprocess_imgs(set_name=X_test, img_size=IMG_SIZE)
X_val_prep = preprocess_imgs(set_name=X_val, img_size=IMG_SIZE)
X_train_prep = preprocess_imgs(set_name=X_train, img_size=IMG_SIZE)

# predict test set
predictions = model.predict(x=X_test_prep)
predictionsVal = model.predict(x=X_val_prep)


"""
    Plot confusion matrix
"""
def plotConfusionMatrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm,interpolation='nearest',cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45) 
    plt.yticks(tick_marks,classes)
    thresh = cm.max() / 2.
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

"""
    Plot validation and training accuracy
    Plot training loss and validation loss
"""
def plotLearningHistory():
    history_dict  = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']
    epochs = range(1, len(loss_values) + 1)


    # Visualize the training process: loss function
    plt.figure()
    plt.plot(epochs, loss_values, 'b', label='Training loss')
    plt.plot(epochs, val_loss_values, 'r', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim([0, 5])
    plt.legend()
    plt.show()


    # Visualize the accuracy
    plt.figure()
    plt.plot(epochs, acc_values, 'b', label='Training acc')
    plt.plot(epochs, val_acc_values, 'r', label='Validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()


plotLearningHistory()


# validate on val set
predictionsVali = model.predict(X_val_prep)
predictionsVali = [1 if x>0.5 else 0 for x in predictionsVali]
accr = accuracy_score(y_val, predictionsVali)
print('Val Accuracy = %.2f' % accr)



# validate on test set
predictionsTes = model.predict(X_test_prep)
predictionsTes = [1 if x>0.5 else 0 for x in predictionsTes]
accr = accuracy_score(y_test, predictionsTes)
print('Test Accuracy = %.2f' % accr)


cm_plot_labels = ['(0,NO)','(1,YES)']
cm = confusion_matrix(y_true=y_test, y_pred=np.round(predictions))
plotConfusionMatrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix- Test data')

cmVal = confusion_matrix(y_true=y_val, y_pred=np.round(predictionsVal))
plotConfusionMatrix(cm=cmVal, classes=cm_plot_labels, title='Confusion Matrix- Validation data')


