# Useful imports
import os
import matplotlib.pyplot as plt
import PIL.Image
import shutil
import random
import cv2
import gc
import math
import time
from keras import applications
from keras.utils.vis_utils import plot_model
from keras.layers import Dense
from keras.models import Model, load_model
from keras.losses import KLDivergence
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras import metrics
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import keras.callbacks
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import h5py
import keras_cv

from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

import pandas as pd

from keras.layers.preprocessing.discretization import summarize
from tensorflow.python.ops.gen_batch_ops import batch
from numpy.lib.function_base import kaiser

import keras.utils as image

import tensorflow_probability as tfp

#Setting up GPU device
name = "CUDA_VISIBLE_DEVICES"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:/user/2022_va_gr07/exit/envs/tf/lib/"
print('os.environ:', os.environ)
#gpus = tf.config.list_physical_devices('GPU')
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        tf.config.set_logical_device_configuration(gpus[0],[tf.config.LogicalDeviceConfiguration(memory_limit=8192)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
print("tf version:", tf.__version__)

demo_mode = False
print("demo_mode:", demo_mode)

"""**MODEL DEFINITION**"""


def build_model(backbone):
  """
  A function to build our model with desired backbone network
    
  :param str backbone : The desired backbone network
  """ 
  base_model = VGGFace(model=backbone, include_top=False, pooling='avg')
  x = base_model.output
  # Creation new top layer block
  predictions = Dense(units=100, activation='softmax')(x)
  # Add new top layer block
  age_model = Model(base_model.input, predictions)
  inputs = keras.Input(shape=(224,224,3)) 
  rescaled = tf.keras.layers.Rescaling(1.0/255)(inputs)
  # Augmentation layers
  #   - Random Flip
  #   - Random Brightness
  #   - Random Shear
  augmented = tf.keras.Sequential([
      tf.keras.layers.RandomFlip("horizontal"),
      tf.keras.layers.RandomBrightness(factor=0.15, value_range=(0.,1.)),
      keras_cv.layers.RandomShear(x_factor=0.1, y_factor=0.1, fill_mode='reflect')
    ])(rescaled)
  # Outputs
  outputs = age_model(augmented)
  model = keras.Model(inputs, outputs)
  return model, age_model, base_model


"""**DATASET LOADING**"""

os.chdir("/mnt/sdc1/2022_va_gr07/dataset") # to read from ssd

if demo_mode:
  num_samples = 236
else:
  num_samples = 575073

num_classes = 100

#Loading image labels from csv file
csv_name = 'training_caip_contest.csv' if not demo_mode else 'demo_training_caip_contest.csv'
data = pd.read_csv(csv_name, nrows=num_samples, header=None)
array_temp = data.to_numpy()
y_dataset = np.delete(array_temp, 0, axis=1).squeeze() # deleting first column

#Accept classes from 1 to n
def one_hot(a, n):
  e = np.eye(n) # Identity matrix n x n
  result=e[a.astype(np.uint8)-1]
  return result

y_dataset_1h = one_hot(y_dataset, num_classes)

print('\nBefore one-hot encoding:')
print(y_dataset)

print('\nAfter one-hot encoding')
print(y_dataset_1h)


class MAEgroup(metrics.Metric):
  """A custom defined MAE metric
    ...
    Attributes
    ----------
    mae : tensor
        the METRIC metric
    K : str
        number of samples on which the MAE metric has been computed
    min_age : int
        the minimum age on which compute the MAE metric
    max_age : int
        the maximum age on which compute the MAE metric
    """

  def __init__(self, name='MAEgroup', min_age=1, max_age=100, **kwargs):
    super(MAEgroup, self).__init__(name=name, **kwargs)
    self.mae = self.add_weight(name='mae', initializer='zeros')
    self.K = self.add_weight(name='K', initializer='zeros', dtype=tf.float32)
    self.min_age = min_age
    self.max_age = max_age

  def update_state(self, y_true, y_pred, sample_weight=None):
    """
    Compute and update the MAE metric
     
    :param ndarray y_true : A ndarray containg the ground truth ages
    :param ndarray y_pred : A ndarray containg the predicted ages 
    """  
    mae_tensor = self.mae.value()
    K_tensor = self.K.value()

    y_true = tf.keras.backend.argmax(y_true, -1) + 1
    # computing y_pred as expectation
    k = tf.linspace(tf.ones(tf.shape(y_true)[0]), tf.ones(tf.shape(y_true)[0])*100, 100, axis=1)
    y_pred = tf.reduce_sum(y_pred * float(k), axis=1)

    y_true_indices = tf.where(tf.math.logical_and(self.min_age <= y_true, y_true <= self.max_age))
    y_true = tf.gather(y_true, y_true_indices)
    y_pred = tf.gather(y_pred, y_true_indices)

    K_current = float(tf.shape(y_true_indices)[0])

    #Compute and update new MAE metric
    if K_current > 0:
      old_sum = mae_tensor * K_tensor
      new_sum = old_sum + float(tf.keras.backend.sum(tf.keras.backend.abs(float(y_true) - y_pred)))
      new_MAEj = new_sum / (K_tensor + float(K_current))

      self.K.assign(K_tensor + float(K_current))
      self.mae.assign(new_MAEj)
    
  def result(self):
    return self.mae

  def reset_state(self):
    self.mae.assign(0)
    self.K.assign(0)



class ComputeAARCallback(keras.callbacks.Callback):
  """A custom defined AAR callback"""

  def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        values = list(logs.values())

        #Compute MAE metric for each one of the age groups on training set
        #{MAE^1 -> 1-10
        # MAE^2 ->  11-20
        # MAE^3 ->  21-30
        # ...
        # MAE^8 ->  71-80}
        mMAE = (logs['MAE^1'] + logs['MAE^2'] + logs['MAE^3'] + logs['MAE^4'] + logs['MAE^5'] + logs['MAE^6']+ logs['MAE^7'] + logs['MAE^8'])/8
        
        #Compute sigma metric on training set       
        sigma_somma_quadrati = pow((logs['MAE^1'] - logs['MAE']), 2) + pow((logs['MAE^2'] - logs['MAE']), 2) +\
                pow((logs['MAE^3'] - logs['MAE']), 2) + pow((logs['MAE^4'] - logs['MAE']), 2) +\
                pow((logs['MAE^5'] - logs['MAE']), 2) + pow((logs['MAE^6'] - logs['MAE']), 2) +\
                pow((logs['MAE^7'] - logs['MAE']), 2) + pow((logs['MAE^8'] - logs['MAE']), 2)
        sigma_somma_divisa = sigma_somma_quadrati/8
        sigma = math.sqrt(sigma_somma_divisa)

        #Compute AAR metric on training set
        AAR = max(0, 5 - mMAE) + max(0, 5 - sigma)

        #Compute MAE metric for each one of the age groups on validation set
        val_mMAE = (logs['val_MAE^1'] + logs['val_MAE^2'] + logs['val_MAE^3'] + logs['val_MAE^4'] + logs['val_MAE^5'] + logs['val_MAE^6']+ logs['val_MAE^7'] + logs['val_MAE^8'])/8

        #Compute sigma metric on validation set
        val_sigma_somma_quadrati = pow((logs['val_MAE^1'] - logs['val_MAE']), 2) + pow((logs['val_MAE^2'] - logs['val_MAE']), 2) +\
                pow((logs['val_MAE^3'] - logs['val_MAE']), 2) + pow((logs['val_MAE^4'] - logs['val_MAE']), 2) +\
                pow((logs['val_MAE^5'] - logs['val_MAE']), 2) + pow((logs['val_MAE^6'] - logs['val_MAE']), 2) +\
                pow((logs['val_MAE^7'] - logs['val_MAE']), 2) + pow((logs['val_MAE^8'] - logs['val_MAE']), 2)
        val_sigma_somma_divisa = val_sigma_somma_quadrati/8
        val_sigma = math.sqrt(val_sigma_somma_divisa)

        #Compute AAR metric on validation set
        val_AAR = max(0, 5 - val_mMAE) + max(0, 5 - val_sigma)

        #Print all the acquired metrics values on epoch end
        print("\nEnd epoch {} of training".format(epoch+1))

        print("MAE^1 = {}\t\t\tMAE^2 = {}\t\t\tMAE^3 = {} \t\t\tMAE^4 = {}\t\t\tMAE^5 = {}\t\t\tMAE^6 = {}\t\t\tMAE^7 = {}\t\t\tMAE^8 = {}"\
              .format(logs['MAE^1'], logs['MAE^2'], logs['MAE^3'], logs['MAE^4'], logs['MAE^5'], logs['MAE^6'], logs['MAE^7'], logs['MAE^8']))

        print("val_MAE^1 = {}\t\t\tval_MAE^2 = {}\t\t\tval_MAE^3 = {} \t\t\tval_MAE^4 = {}\t\t\tval_MAE^5 = {}\t\t\tval_MAE^6 = {}\t\t\tval_MAE^7 = {}\t\t\tval_MAE^8 = {}"\
              .format(logs['val_MAE^1'], logs['val_MAE^2'], logs['val_MAE^3'], logs['val_MAE^4'], logs['val_MAE^5'], logs['val_MAE^6'], logs['val_MAE^7'], logs['val_MAE^8']))

        print("mMAE = {}\t\t\tsigma = {}\t\t\tAAR = {}".format(mMAE, sigma, AAR))

        print("val_mMAE = {}\t\t\tval_sigma = {}\t\t\tval_AAR = {}".format(val_mMAE, val_sigma, val_AAR))

  def on_epoch_begin(self, epoch, logs=None):
    print()

if demo_mode:
  os.chdir("demo_training_caip_contest")
else:
  os.chdir("training_caip_contest")

model_name = 'A3MoreDense'


"""**DATASET SPLITTING**"""


DATASET_SPLIT = (0.8, 0.2)
batch_size = 32

entries = os.listdir('/user/2022_va_gr07/AVProject/model_saves/')

assert 'dataset_split_indexes.h5' in entries

#Load old indexes if any have been found
f1 = h5py.File('/user/2022_va_gr07/AVProject/model_saves/dataset_split_indexes.h5', 'r+')
idxs = np.copy(f1.get('idxs'))
f1.close()
print("old idxs found")

class CustomDataSequence(tf.keras.utils.Sequence):
    """
    Custom data sequence to load training and validation images
    ...
    Attributes
    ----------
    shuffled_indexes : ndarray
        A ndarray containg the passed dataset indexes
    batch_size : int
        The batch size (default=32)
    """ 

    def __init__(self, shuffled_indexes, batch_size=32):
        self.batch_size = batch_size
        self.__batch_index = 0
        self.shuffled_indexes = shuffled_indexes
        np.random.shuffle(self.shuffled_indexes)

    def __len__(self):
        return int(np.floor(len(self.shuffled_indexes)/self.batch_size))

    def next(self):
        return self.__next__()

    def __next__(self):
        return self[self.__batch_index]

    def __getitem__(self, batch_index):
      """
      Take n random images, where n is the batch_size, and load them into a batch
      """ 
      start = batch_index*self.batch_size
      stop = start + self.batch_size
      if stop > len(self.shuffled_indexes):
          raise StopIteration
      this_batch_indexes = self.shuffled_indexes[start:stop]
 
      x = np.array([image.img_to_array(image.load_img(str(i)+".jpg", target_size = (224, 224), interpolation = "bilinear"))  for i in this_batch_indexes])
      y = np.array([y_dataset_1h[i] for i in this_batch_indexes])
      return (x, y)

    def on_epoch_end(self):
        np.random.shuffle(self.shuffled_indexes)


#Get train and validation indexes
train_idxs = idxs[:int(num_samples*DATASET_SPLIT[0])]
validation_idxs = idxs[int(num_samples*DATASET_SPLIT[0]):]

#Get train and validation sequences
train_sequence = CustomDataSequence(train_idxs, batch_size = batch_size)
validation_sequence = CustomDataSequence(validation_idxs, batch_size = batch_size)

tfd = tfp.distributions

def representation_loss(y_true, y_pred):
  """
  A custom defined loss for the representation stage
    
  :param ndarray y_true : A ndarray containg the ground truth ages
  :param ndarray y_pred : A ndarray containg the predicted ages 
  """ 

  true_age = float(tf.math.argmax(y_true, axis=-1) + 1)

  d = tfd.Normal(loc=true_age, scale=float(tf.ones([tf.shape(y_true)[0]])) )

  d_samples = d.prob([[x] for x in range(1,101,1)])
  d_samples = tf.transpose(d_samples)

  kl = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
  L_ld_new = kl(d_samples, y_pred)

  k = tf.linspace(tf.ones(tf.shape(y_true)[0]), tf.ones(tf.shape(y_true)[0])*100, 100, axis=1)

  y_estimated = tf.reduce_sum(y_pred * float(k), axis=1)

  L_er = tf.keras.backend.abs(true_age - y_estimated)

  lamb = 1

  L = L_ld_new + (lamb * L_er)
  
  return L

# To not show warnings caused by preprocessing layers
tf.get_logger().setLevel('ERROR')

class MonitorValAAR(keras.callbacks.Callback):
  """
  A custom defined ARR for validation set on epoch end
  """ 

  def __init__(self, filepath, patience=3):
    super(keras.callbacks.Callback, self).__init__()
    self._currentValAAR = 0
    self.filepath = filepath
    self.patience = patience
    self._count = 0
    self._endCount = 0
  
  def on_epoch_end(self, epoch, logs={}):
    #Compute mMAE for validation set
    val_mMAE = (logs['val_MAE^1'] + logs['val_MAE^2'] + logs['val_MAE^3'] + logs['val_MAE^4'] + logs['val_MAE^5'] + logs['val_MAE^6']+ logs['val_MAE^7'] + logs['val_MAE^8'])/8
    
    #Compute sigma for validation set
    val_sigma_somma_quadrati = pow((logs['val_MAE^1'] - logs['val_MAE']), 2) + pow((logs['val_MAE^2'] - logs['val_MAE']), 2) +\
            pow((logs['val_MAE^3'] - logs['val_MAE']), 2) + pow((logs['val_MAE^4'] - logs['val_MAE']), 2) +\
            pow((logs['val_MAE^5'] - logs['val_MAE']), 2) + pow((logs['val_MAE^6'] - logs['val_MAE']), 2) +\
            pow((logs['val_MAE^7'] - logs['val_MAE']), 2) + pow((logs['val_MAE^8'] - logs['val_MAE']), 2)
    val_sigma_somma_divisa = val_sigma_somma_quadrati/8
    val_sigma = math.sqrt(val_sigma_somma_divisa)

    #Compute AAR for validation set
    val_AAR = max(0, 5 - val_mMAE) + max(0, 5 - val_sigma)

    #Checks for AAR improvements
    if(val_AAR > self._currentValAAR):
      #Overrides the model saved weights with the ones which achieved a better AAR
      print("Epoch {}: val_AAR improved from {} to {}".format(epoch+1, self._currentValAAR, val_AAR))
      model.save_weights(self.filepath)
      print("Saving model to {}".format(self.filepath))
      self._currentValAAR = val_AAR
      self._count = 0
    else:
      print("Epoch {}: val_AAR did not improve from {}".format(epoch+1, self._currentValAAR))
      self._count = self._count + 1
      if(val_AAR != 0 and self._count == self.patience):
        print("val_AAR did not improve for {} consecutive epochs".format(self.patience))
        self._count = 0
        self._endCount = self._endCount + 1
        if(self._endCount < 3):
          #If AAR improves on the last 3 epochs decreases the learning rate
          lr = float(keras.backend.get_value(self.model.optimizer.lr))
          new_lr = lr*0.2
          keras.backend.set_value(self.model.optimizer.lr, new_lr)
          print("new lr set to {}".format(float(keras.backend.get_value(self.model.optimizer.lr))))
        else:
          #If AAR didn't improve for 3 epochs stops the training procedure
          self.model.stop_training = True
          print("End training\n")
    
# Compute evaluation MAE for classification
model, age_model, base_model = build_model("resnet50")
model.load_weights(filepath = '/user/2022_va_gr07/AVProject/model_saves/' +'A3_repr' +'.h5')
print('Model ' +'/user/2022_va_gr07/AVProject/model_saves/' +'A3_repr' +'.h5' +' correctly loaded')

model.compile(loss=representation_loss,
              optimizer=SGD(learning_rate = 0.005),                           
              metrics=[MAEgroup("MAE")]
              )

print("EVALUATING REPR MODEL ON VALIDATION TO COMPUTE val_MAE")

_, val_repr_mae = model.evaluate(validation_sequence, 
                                 workers = 32,
                                 max_queue_size = 10,
                                 use_multiprocessing = True,
                                 verbose=1)

print("val_repr_mae ={}".format(val_repr_mae))


"""**CLASSIFICATION STAGE**"""


class ClassBalancedSequence(tf.keras.utils.Sequence):
    """
    Custom data sequence to load images, choising them with 
    the same probability from all the age groups
    """ 

    def __init__(self, shuffled_indexes, batch_size=32):
        self.batch_size = batch_size
        self.__batch_index = 0
        self._ages = []

        indexes_ages = {}

        for i in range (1,82,1):
          indexes_ages[i] = []

        np.random.shuffle(shuffled_indexes)

        for index in shuffled_indexes:
          age = y_dataset[index]
          indexes_ages[age].append(index)

        for i in range (1,82,1):
          if len(indexes_ages[i]) == 0:
            del indexes_ages[i]
          else:
            self._ages.append(i)

        self.indexes_ages = indexes_ages
        self.shuffled_indexes = shuffled_indexes
        print(self._ages)
        for age in indexes_ages:
          print("AGE: " + str(age) + " > ", end='')
          count = 0
          for i in indexes_ages[age]:
            count+=1
          print(count)        

    def __len__(self):
        return int(np.floor(len(self.shuffled_indexes)/self.batch_size))

    def next(self):
        return self.__next__()

    def __next__(self):
        return self[self.__batch_index]

    def __getitem__(self, batch_index):
      """
      Take n random images, where n is the batch_size, and load them into a batch,
      ensuring an equal probability for each age group
      """ 
      start = batch_index*self.batch_size
      stop = start + self.batch_size
      if stop > len(self.shuffled_indexes):
          raise StopIteration
      classes = random.choices(self._ages, k=batch_size)
      x = []
      y = []
      for cl in classes:
        rng_index = random.choice(self.indexes_ages[cl])
        x.append(image.img_to_array(image.load_img(str(rng_index)+".jpg", target_size = (224, 224), interpolation = "bilinear")))
        y.append(y_dataset_1h[rng_index])
      x = np.array(x)    
      y = np.array(y)      
      return (x, y)

    def on_epoch_end(self):
      np.random.shuffle(self.shuffled_indexes)

def classification_loss(y_true, y_pred):
  """
  A custom defined loss for the classification stage
    
  :param ndarray y_true : A ndarray containg the ground truth ages
  :param ndarray y_pred : A ndarray containg the predicted ages 
  """ 

  true_age = float(tf.math.argmax(y_true, axis=-1) + 1)

  d = tfd.Normal(loc=true_age, scale=float(tf.ones([tf.shape(y_true)[0]])) )

  d_samples = d.prob([[x] for x in range(1,101,1)])
  d_samples = tf.transpose(d_samples)

  kl = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
  L_ld_new = kl(d_samples, y_pred)

  k = tf.linspace(tf.ones(tf.shape(y_true)[0]), tf.ones(tf.shape(y_true)[0])*100, 100, axis=1)

  y_estimated = tf.reduce_sum(y_pred * float(k), axis=1)

  L_er = tf.keras.backend.abs(true_age - y_estimated)

  val_repr_mae_tensor = tf.ones([tf.shape(y_true)[0]])*val_repr_mae

  L = L_ld_new + tf.keras.backend.square(L_er - val_repr_mae_tensor)
  
  return L


EPOCHS = 100

classification_train_sequence = ClassBalancedSequence(train_idxs, batch_size = batch_size)

train_aft = len(age_model.layers)
for layer in age_model.layers[:train_aft-1]:
  layer.trainable = False


x = base_model.output
# Creating new top layer block and adding the 3 dense layers
x1 = Dense(units=100, activation='relu')(x)
x2 = Dense(units=100, activation='relu')(x1)
predictions = Dense(units=100, activation='softmax')(x2)
# Add new top layer block
age_model = Model(base_model.input, predictions)
inputs = keras.Input(shape=(224,224,3)) 
rescaled = tf.keras.layers.Rescaling(1.0/255)(inputs)
# Augmentation layers
augmented = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomBrightness(factor=0.15, value_range=(0.,1.)),
    keras_cv.layers.RandomShear(x_factor=0.1, y_factor=0.1, fill_mode='reflect')
  ])(rescaled)
# Outputs
outputs = age_model(augmented)
model = keras.Model(inputs, outputs)

model.summary()

print("CLASSIFICATION STAGE")

model.compile(loss=classification_loss,
              optimizer=SGD(learning_rate = 0.00015),  
              metrics=[
                  MAEgroup("MAE"),
                  MAEgroup("MAE^1", min_age=1, max_age=10),
                  MAEgroup("MAE^2", min_age=11, max_age=20),
                  MAEgroup("MAE^3", min_age=21, max_age=30),
                  MAEgroup("MAE^4", min_age=31, max_age=40),
                  MAEgroup("MAE^5", min_age=41, max_age=50),
                  MAEgroup("MAE^6", min_age=51, max_age=60),
                  MAEgroup("MAE^7", min_age=61, max_age=70),
                  MAEgroup("MAE^8", min_age=71, max_age=100)
                  ],
              )
  
h_class = model.fit(classification_train_sequence, 
            workers = 32,
            max_queue_size = 10,
            use_multiprocessing = True,
            validation_data=validation_sequence, 
            epochs=EPOCHS, 
            callbacks=[
                ComputeAARCallback(),
                MonitorValAAR('/user/2022_va_gr07/AVProject/model_saves/' +model_name +'_class' +'.h5', patience=3)
                ],
            verbose=1)