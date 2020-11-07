# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 22:50:03 2020

@author: rasto
"""
import os
import cv2
import itertools
import numpy as np
#import seaborn as sns
#from PIL import Image
from tqdm import tqdm
import splitfolders as splitter
import matplotlib.pyplot as plt


import tensorflow as tf

from keras.optimizers import adam
#from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Activation, Flatten, Dense
from keras import backend as K

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# split the binary data into train/test/valid folers by ratio, only when needed
#splitter.ratio('/home/rastislav/Desktop/bp/brain_tumor_dataset/dataset-bigger/', output="/home/rastislav/Desktop/bp/brain_tumor_dataset/dataset-bigger-spl/", seed=1337, ratio=(.7, 0.2,0.1))


#TRAIN_DIR='/home/rastislav/Desktop/bp/brain_tumor_dataset/dataset/train/'
#TEST_DIR='/home/rastislav/Desktop/bp/brain_tumor_dataset/dataset/test/'
#VAL_DIR='/home/rastislav/Desktop/bp/brain_tumor_dataset/dataset/val/'
TRAIN_DIR='/home/rastislav/Desktop/bp/brain_tumor_dataset/dataset-bigger-spl/train/'
TEST_DIR='/home/rastislav/Desktop/bp/brain_tumor_dataset/dataset-bigger-spl/test/'
VAL_DIR='/home/rastislav/Desktop/bp/brain_tumor_dataset/dataset-bigger-spl/val/'
IMG_SIZE=(128,128)
EPOCHS=30
RANDOM_SEED=123


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
NUM_CLASSES=1



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

#plt.imshow(X_train[81])
#print(y_train[81])


######################################## Creating model ##########################
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

if K.image_data_format() == 'channels_first':
    input_shape = (3, IMG_SIZE[0], IMG_SIZE[1])
else:
    input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)

model = Sequential()

#model.add(Conv2D(32, (7,7), input_shape=input_shape))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(32, (5,5), padding="same", activation="relu"))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



batch_size = 12

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.05,
        height_shift_range=0.05,
        rescale=1./255,
        shear_range=0.2,
        brightness_range=[0.1, 1.5],
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        )


# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(
    rescale=1./255
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
validation_generator = test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=batch_size,
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
    patience=6
)

model.summary()



history = model.fit_generator(
        train_generator,
   #     steps_per_epoch=120,
        epochs=EPOCHS,
        validation_data=validation_generator,
    #    validation_steps=25,
        callbacks=[es]
        )

model.save_weights('first_try.h5')  # always save your weights after training or during training



# predict test set
predictions = model.predict(x=X_test)
predictionsVal = model.predict(x=X_val)


"""
    Plot confusion matrix
"""
def plotConfusionMatrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    count_wrong_predict = 0
    for i in range(0,predictions.size):
        if (predictions[i] != y_test[i]):
            count_wrong_predict+=1
    print("Wrong predictions: ", count_wrong_predict, "  of total: ", len(y_test) ,". Accuracy: ", (len(y_test) -count_wrong_predict)/len(y_test))
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
    plt.ylim([0, 2])
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


cm_plot_labels = ['NO','YES ']
cm = confusion_matrix(y_true=y_test, y_pred=np.round(predictions))
plotConfusionMatrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix- Test data')

cmVal = confusion_matrix(y_true=y_val, y_pred=np.round(predictionsVal))
plotConfusionMatrix(cm=cmVal, classes=cm_plot_labels, title='Confusion Matrix- Validation data')

