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


# In[8]:


# Basic CNN model -1D - kernel_size is required
model = models.Sequential()
model.add(layers.Conv1D(32, kernel_size=2, activation='relu', input_shape=(121, 1)))
model.add(layers.MaxPooling1D(2))


# In[9]:


model.add(layers.Flatten())
model.add(layers.Dense(23, activation='softmax'))
model.summary()


# In[10]:


from tensorflow.keras.utils import get_file

try:
    path = get_file('kddcup.data_10_percent.gz', origin='http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz')
except:
    print('Error downloading')
    raise

print(path) 
df = pd.read_csv(path, header=None)

print("Read {} rows.".format(len(df)))
df.dropna(inplace=True,axis=1) 

df[0:5]


# In[11]:


df.columns = [
    'duration',
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'land',
    'wrong_fragment',
    'urgent',
    'hot',
    'num_failed_logins',
    'logged_in',
    'num_compromised',
    'root_shell',
    'su_attempted',
    'num_root',
    'num_file_creations',
    'num_shells',
    'num_access_files',
    'num_outbound_cmds',
    'is_host_login',
    'is_guest_login',
    'count',
    'srv_count',
    'serror_rate',
    'srv_serror_rate',
    'rerror_rate',
    'srv_rerror_rate',
    'same_srv_rate',
    'diff_srv_rate',
    'srv_diff_host_rate',
    'dst_host_count',
    'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'outcome'
]


def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)

#  Now encode the feature vector

# encode_numeric_zscore(df, 'duration')
encode_text_dummy(df, 'protocol_type')
encode_text_dummy(df, 'service')
encode_text_dummy(df, 'flag')
# encode_numeric_zscore(df, 'src_bytes')
# encode_numeric_zscore(df, 'dst_bytes')
encode_text_dummy(df, 'land')

encode_text_dummy(df, 'logged_in')

encode_text_dummy(df, 'is_host_login')
encode_text_dummy(df, 'is_guest_login')

df.dropna(inplace=True,axis=1)
x_columns = df.columns.drop('outcome')
x = df[x_columns].values
dummies = pd.get_dummies(df['outcome']) # Classification
outcomes = dummies.columns
num_classes = len(outcomes)
y = dummies.values


# In[12]:


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


# In[13]:


test_loss, test_acc = model.evaluate(x_test, y_test)

print('Test loss:', test_loss)
print('Test accuracy:', test_acc)
# with Zscores- training and testing accuracy 99.++
# with Zscores- training and testing loss 0.023++
# without Zscores- training accuracy and loss 0.9905, 0.0371
# without Zscores- training and testing loss 0.991878, 0.0341


# In[ ]:




