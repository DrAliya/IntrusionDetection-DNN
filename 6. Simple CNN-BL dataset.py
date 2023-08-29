#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras import layers
from keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
#CNN
from keras.models import Sequential,Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import keras
keras.__version__


# In[13]:


# Basic CNN model -1D - kernel_size is required
model = models.Sequential()
model.add(layers.Conv1D(32, kernel_size=2, activation='relu', input_shape=(9, 1)))
model.add(layers.MaxPooling1D(2))


# In[14]:


model.add(layers.Flatten())
model.add(layers.Dense(2, activation='softmax'))
model.summary()


# In[19]:


import pandas as pd

df = pd.read_csv(r"C:\Users\Admin\Desktop\Year2019-20\PhD Thesis\DATASET\Bluetooth\MergedFinal1.csv", header=None)

print("Read {} rows.".format(len(df)))

df.dropna(inplace=True,axis=1) # For now, just drop NA's (rows with missing values)

df.columns = [
     'Frame_length_stored_into_the_capture_file_per_100msec',
     'Length_per_100msec',
   'L2CAP_count_per_100msec',
   'HCI_ACL_count_per_100msec',
  'HCI_EVT_count_per_100msec',
   'Received_count_per_100msec',
    'Sent_count_per_100msec',
    'Command_Complete_count_per_100msec',
    'Disconnect_complete_count_per_100msec',
     'outcome'
]

def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()

    if sd is None:
        sd = df[name].std()

    df[name] = (df[name] - mean) / sd
    
encode_numeric_zscore(df, 'Frame_length_stored_into_the_capture_file_per_100msec')
encode_numeric_zscore(df, 'Length_per_100msec')
encode_numeric_zscore(df, 'L2CAP_count_per_100msec')
encode_numeric_zscore(df, 'HCI_ACL_count_per_100msec')
encode_numeric_zscore(df, 'HCI_EVT_count_per_100msec')
encode_numeric_zscore(df, 'Received_count_per_100msec')
encode_numeric_zscore(df, 'Sent_count_per_100msec')
encode_numeric_zscore(df, 'Command_Complete_count_per_100msec')
encode_numeric_zscore(df, 'Disconnect_complete_count_per_100msec')


# In[20]:


x_columns = df.columns.drop('outcome')
x = df[x_columns].values
dummies = pd.get_dummies(df['outcome']) # Classification
outcomes = dummies.columns
num_classes = len(outcomes)
y = dummies.values


# In[21]:


# Create a test/train split.  25% test
# Split into train/test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
batch_size = 100
epochs = 5

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1
          )


# In[23]:


test_loss, test_acc = model.evaluate(x_test, y_test)

print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

#without Z-score conversion = training accuracy 0.7933, Training Loss:0.5208
#without Z-score conversion = testing accuracy 0.79943, Training Loss:0.50028
#with Z-score conversion = training accuracy 0.9239, Training Loss:0.4998
#with Z-score conversion = testing accuracy 0.9500, Training Loss:0.455197


# In[ ]:




