#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Imports
import zipfile
from pathlib import Path
import os
from os import listdir
from os.path import isfile, join
import os, re, math, json, shutil, pprint
import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import IPython.display as display
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix
from PIL import Image
import keras_tuner as kt


# In[2]:


#Global variables
BATCH_SIZE = 128
EPOCHS = 20


# In[3]:


# lr decay function
def lr_decay(epoch):
  return 0.01 * math.pow(0.725, epoch)

def load_dataset(dataList, labelList):
    imagedataset = tf.data.Dataset.from_tensor_slices(dataList)
    labelsdataset = tf.data.Dataset.from_tensor_slices(labelList)
    dataset = tf.data.Dataset.zip((imagedataset, labelsdataset))
    return dataset 

#Load the data as dataset from arrays and configure those datasets
def get_training_dataset():
    dataset = load_dataset(trainData, trainLabels)
    dataset = dataset.cache()
    dataset = dataset.shuffle(28700, reshuffle_each_iteration=True)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def get_validation_dataset():
    dataset = load_dataset(valData, valLabels)
    dataset = dataset.take(3580).cache()
    dataset = dataset.batch(3580, drop_remainder=True) # 5000 items in eval dataset, all in one batch
    return dataset

def get_test_dataset():
    dataset = load_dataset(testData, testLabels)
    dataset = dataset.take(3580).cache() # this small dataset can be entirely cached in RAM, for TPU this is important to get good performance from such a small dataset
    dataset = dataset.batch(3580, drop_remainder=True) # 5000 items in eval dataset, all in one batch
    return dataset

#Convert a dataframe to a numpy array
def dfToArray(df):
    df['pixels'] = df['pixels'].apply(lambda values: [int(i) for i in values.split()])
    data = np.array(df['pixels'].tolist(), dtype='int32').reshape(-1,48, 48,1) 
    labels = to_categorical(df['emotion'], 7)  
    return (data, labels)

#Model creation, this is the structure of the CNN neural network
def create_model():
    model = tf.keras.Sequential(
        [
        #1 Convolutionnal layer N°1, input 48*48, 12 filters, kernel_size = 3
        tf.keras.layers.Conv2D(12, kernel_size=3, input_shape=(48,48,1), data_format='channels_last'),
        tf.keras.layers.BatchNormalization(center=True, scale=False),
        tf.keras.layers.Activation('relu'),

        #2 Convolutionnal layer N°2, 24 filters, kernel_size = 3 | Pooling after activation
        tf.keras.layers.Conv2D(24, kernel_size=3, padding='same'),
        tf.keras.layers.BatchNormalization(center=True, scale=False),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),

        #3 Convolutionnal layer N°3, 36 filters, kernel_size = 3
        tf.keras.layers.Conv2D(36, kernel_size=3, padding='same'),
        tf.keras.layers.BatchNormalization(center=True, scale=False),
        tf.keras.layers.Activation('relu'),

        #4 Convolutionnal layer N°4, 48 filters, kernel_size = 3 | Pooling after activation
        tf.keras.layers.Conv2D(48, kernel_size=3, padding='same'),
        tf.keras.layers.BatchNormalization(center=True, scale=False),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),

        #5 Convolutionnal layer N°5, 64 filters, kernel_size = 3
        tf.keras.layers.Conv2D(64, kernel_size=3, padding='same'),
        tf.keras.layers.BatchNormalization(center=True, scale=False),
        tf.keras.layers.Activation('relu'),
    
        #6 Convolutionnal layer N°6, 86 filters, kernel_size = 3| Pooling after activation
        tf.keras.layers.Conv2D(86, kernel_size=3, padding='same'),
        tf.keras.layers.BatchNormalization(center=True, scale=False),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),

        tf.keras.layers.Flatten(), #Flatten to convert to dense layers

        #7 Dense layer N°1, 500 neurons
        tf.keras.layers.Dense(500),
        tf.keras.layers.BatchNormalization(center=True, scale=False),
        tf.keras.layers.Activation('relu'),

        #8 Dense layer N°2, 250 neurons
        tf.keras.layers.Dense(250),
        tf.keras.layers.BatchNormalization(center=True, scale=False),
        tf.keras.layers.Activation('relu'),

        #9 Dense layer N°3, 125 neurons
        tf.keras.layers.Dense(125),
        tf.keras.layers.BatchNormalization(center=True, scale=False),
        tf.keras.layers.Activation('relu'),

        #10 Final dense layer, 7 neurons, one for each class
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    return model


# In[90]:


#Labels corresponding to the one hot encoding of the emotions
emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

#Many plotting functions, most of them stolen from the internet
def setup_axe(axe,df,title):
    df['emotion'].value_counts(sort=True).plot(ax=axe, kind='bar', rot=0)
    emotionLabels = []
    for i in axe.get_xticklabels():
        emotionLabels.append(emotion_labels[int(i.get_text())])
    axe.set_xticklabels(emotionLabels)
    axe.set_xlabel("Emotions")
    axe.set_ylabel("Number")
    axe.set_title(title)
    
    # set individual bar lables using above list
    for i in axe.patches:
        # get_x pulls left or right; get_height pushes up or down
        axe.text(i.get_x()-.05, i.get_height()+120,                 str(round((i.get_height()), 2)), fontsize=14, color='dimgrey',
                    rotation=0)
        
# utility to display multiple rows of digits, sorted by unrecognized/recognized status
def display_top_unrecognized(digits, predictions, labels, n, lines):
  idx = np.argsort(predictions==labels) # sort order: unrecognized first
  for i in range(lines):
    display_digits(digits[idx][i*n:(i+1)*n], predictions[idx][i*n:(i+1)*n], labels[idx][i*n:(i+1)*n],
                   "{} sample validation digits out of {} with bad predictions in red and sorted first".format(n*lines, len(digits)) if i==0 else "", n)
    
# utility to display a row of digits with their predictions
def display_digits(digits, predictions, labels, title, n):
  fig = plt.figure(figsize=(24,24))
  digits = np.reshape(digits, [n, 48, 48])
  digits = np.swapaxes(digits, 0, 1)
  digits = np.reshape(digits, [48, 48*n])
  plt.yticks([])
  emotionLabels = []
  for i in predictions:
    emotionLabels.append(emotion_labels[i])
  plt.xticks([48*x+24 for x in range(n)], emotionLabels)
  plt.grid(b=None)
  for i,t in enumerate(plt.gca().xaxis.get_ticklabels()):
    if predictions[i] != labels[i]: t.set_color('red') # bad predictions in red
  plt.imshow(digits,cmap='gray')
  plt.grid(None)
  plt.title(title)
  
    
def dataset_to_numpy_util(training_dataset, validation_dataset, N):
  
  # get one batch from each: 10000 validation digits, N training digits
  batch_train_ds = training_dataset.unbatch().batch(N)
  
  # eager execution: loop through datasets normally
  if tf.executing_eagerly():
    for validation_digits, validation_labels in validation_dataset:
      validation_digits = validation_digits.numpy()
      validation_labels = validation_labels.numpy()
      break
    for training_digits, training_labels in batch_train_ds:
      training_digits = training_digits.numpy()
      training_labels = training_labels.numpy()
      break
    
  else:
    v_images, v_labels = validation_dataset.make_one_shot_iterator().get_next()
    t_images, t_labels = batch_train_ds.make_one_shot_iterator().get_next()
    # Run once, get one batch. Session.run returns numpy results
    with tf.Session() as ses:
      (validation_digits, validation_labels,
       training_digits, training_labels) = ses.run([v_images, v_labels, t_images, t_labels])
  
  # these were one-hot encoded in the dataset
  validation_labels = np.argmax(validation_labels, axis=1)
  training_labels = np.argmax(training_labels, axis=1)
  
  return (training_digits, training_labels,
          validation_digits, validation_labels)
    
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots(figsize=(12,6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def plot_learning_rate(lr_func, epochs):
  xx = np.arange(epochs+1, dtype=np.float)
  y = [lr_decay(x) for x in xx]
  fig, ax = plt.subplots(figsize=(9, 6))
  ax.set_xlabel('epochs')
  ax.set_title('Learning rate\ndecays from {:0.3g} to {:0.3g}'.format(y[0], y[-2]))
  ax.minorticks_on()
  ax.grid(True, which='major', axis='both', linestyle='-', linewidth=1)
  ax.grid(True, which='minor', axis='both', linestyle=':', linewidth=0.5)
  ax.step(xx,y, linewidth=3, where='post')


# In[5]:


#Path configuration
currentPath = Path(os.getcwd())
mainPath = ""

if os.path.exists(currentPath / "files"):
    mainPath = Path(currentPath)
else:
    mainPath = Path(currentPath).parent

srcPath = mainPath / "src"
filesPath = mainPath / "files"


# In[6]:


#Extract the dataset from it's compressed .zip
if not os.path.exists(filesPath / "fer2013.csv"):
    if os.path.exists(filesPath / "compressedDataset.zip"):
        with zipfile.ZipFile(filesPath / "compressedDataset.zip", 'r') as zip_ref:
            zip_ref.extractall(filesPath)
    else:
        print("Could not find compressed dataset, exiting.")
        exit(0)
else:
    print("Dataset already exists, proceeding.")


# In[92]:


#Read the csv
data = pd.read_csv(filesPath / "fer2013.csv")

#Split the dataframe into the test, validation and test sets with a 80/10/10 repartition respectively
trainDf = data.sample(frac=0.80,random_state=200)
remain = data.drop(trainDf.index)

valDf = remain.sample(frac=0.5,random_state=200)
remain = remain.drop(valDf.index)

testDf = remain.sample(frac=1.0,random_state=200)


# In[93]:


#Dataframe to array conversion with reshaping to a 48*48 matrix for the data and one hot encoding of the labels
trainData, trainLabels = dfToArray(trainDf)
valData, valLabels = dfToArray(valDf)
testData, testLabels = dfToArray(testDf)

#Array to dataset conversion
training_dataset = get_training_dataset()
validation_dataset = get_validation_dataset()
test_dataset = get_test_dataset()


# In[94]:


plot_learning_rate(lr_decay, EPOCHS)


# In[95]:


#Plot bar graphs of the population of each set
fig, axes = plt.subplots(1,3, figsize=(20,8), sharey=True)
setup_axe(axes[0],trainDf,'train')
setup_axe(axes[1],valDf,'validation')
setup_axe(axes[2],testDf,'test')
plt.show()


# In[96]:


N = 12 #More display
(training_digits, training_labels,
 validation_digits, validation_labels) = dataset_to_numpy_util(training_dataset, validation_dataset, N)
display_digits(training_digits, training_labels, training_labels, "training digits and their labels", N)
display_digits(validation_digits[:N], validation_labels[:N], validation_labels[:N], "validation digits and their labels", N)


# In[12]:


#Tried to use Keras Tuner to find the best hyperparameters but it crashes for some reasons
'''
tuner = kt.RandomSearch(
    create_model,
    objective='val_accuracy',
    max_trials=1,
    directory='tuning')

tuner.search(trainData, trainLabels, epochs=1, validation_data=validation_dataset)
best_model = tuner.get_best_models()[0]
'''


# In[13]:


#Create the keras model
model = create_model()

# Compute the step by epoch from the training dataframe size
steps_per_epoch = len(trainDf)//BATCH_SIZE 

#Learning rate callback
lr_decay_callback = tf.keras.callbacks.LearningRateScheduler(lr_decay, verbose=True)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="cp.ckpt",
                                                 save_weights_only=True,
                                                 verbose=1)

#Early stop callback, stops the model early if no progress is made anymore
es_callback = EarlyStopping(monitor='val_loss', patience = 5, mode = 'min', restore_best_weights=True)

#Slightly modify the images in order to increase diversity and prevent overfitting
imageGenerator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)

#Feed our dataset into the image generator we previously made
dataFeed = imageGenerator.flow(trainData, trainLabels,batch_size=BATCH_SIZE)


# In[14]:


#Check if a checkpoint already exists. Either load it up or generate a new fit
checkpoint_path = "cp.ckpt"
onlyfiles = [f for f in listdir(os.getcwd()) if isfile(join(os.getcwd(), f)) and checkpoint_path == f[:len(checkpoint_path)]]

history = None
if len(onlyfiles) != 0:
    print("Loading already existing model")
    model.load_weights(checkpoint_path)
else:
    print("Fitting a new model")
    #Train the model using the training data and the validation data. Has callbacks for learning rate, checkpoint and early stop
    history = model.fit(dataFeed, steps_per_epoch=steps_per_epoch, epochs=EPOCHS,
                    validation_data=validation_dataset, validation_steps=1, callbacks=[lr_decay_callback,cp_callback,es_callback])


# In[15]:


# Again stolen from the internet,
if history != None:
    fig, axes = plt.subplots(1,2, figsize=(18, 6))
    # Plot training & validation accuracy values
    axes[0].plot(history.history['accuracy'])
    axes[0].plot(history.history['val_accuracy'])
    axes[0].set_title('Model accuracy')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(['Train', 'Validation'], loc='upper left')

   
    axes[1].plot(history.history['loss'])
    axes[1].plot(history.history['val_loss'])
    axes[1].set_title('Model loss')
    axes[1].set_ylabel('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(['Train', 'Validation'], loc='upper left')
    plt.show()


# In[53]:


#Display some of the unrecognized images to get an idea of what the model is getting wrong
probas = model.predict(test_dataset, verbose=1)
predicted_labels = np.argmax(probas, axis=1)
display_top_unrecognized(validation_digits,predicted_labels,validation_labels,24,7)


# In[54]:


#Get the accuracy on the test dataset as a final result. This dataset has never been seen before by the model
test_true = np.argmax(testLabels, axis=1)
test_pred = np.argmax(model.predict(testData), axis=1)
print("Accuracy on the test set",accuracy_score(test_true, test_pred))


# In[55]:


#Plot the confusion matrix to see what classes the model has trouble distinguishing
plot_confusion_matrix(test_true, test_pred, classes=emotion_labels, normalize=True, title='Normalized confusion matrix')


# In[57]:


#Recover the images for the ImagesProjet folder
onlyfiles = [f for f in listdir(mainPath / "ImagesProjet") if isfile(join(mainPath / "ImagesProjet", f))]

fileNames = []
customData = [] 
customLabels = [] 

#Convert those images to greyscale and downscale them to 48*48
for i in onlyfiles:
    name = "Grey_" + i
    img = Image.open(mainPath / "ImagesProjet" / i).convert('L')
    img = img.resize((48,48))
    fileNames.append(i)

    customData.append(np.asarray(img).reshape(48,48))
    customLabels.append(tf.one_hot(tf.reshape(3,[]),7))
    img.save(mainPath / "ImagesProjet" / "greyImages" / name)

#Reshape those arrays to fit the model inputs
npcustomData = np.asarray(customData).reshape(-1,48,48)
npcustomLabels = np.asarray(customLabels).reshape(-1,7)


# In[58]:


#Get the prediction of the model on those custom images and display it
probas = model.predict(npcustomData, verbose=1)
predicted_labels = np.argmax(probas, axis=1)
display_digits(npcustomData, predicted_labels, predicted_labels, "¨Predictions on custom picked images",len(customData))

