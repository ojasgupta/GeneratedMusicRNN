# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 16:43:35 2017

@author: arclab
"""
#%% package imports
import numpy as np
import re
import itertools
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Activation
from keras.callbacks import ModelCheckpoint
from random import randint
import matplotlib.pyplot as plt
#%% function defintions

def loadMusic(file):
    f = open(file, 'r')
    text = f.read()
    words = re.split(r"(\s+)", text)
    new_words = [x for x in words if (x != '<start>' and x != '<end>')] # get rid of <start> and <end>
    data = []
    [data.append(list(w)) for w in new_words]
    merged_data = np.asarray(list(itertools.chain.from_iterable(data)))  
#    s = set(merged_data)
#    char_int = [ch:i for i,ch in enumerate(s)] # encode characters to integers
#    int_char = [i:ch for i,ch in enumerate(s)] # encode integers to characters
    encoded_data = [ord(x) for x in merged_data] # assing ascii labels to characters
    return encoded_data

def prepareData(data, t):
	dataX, dataY = [], []
	for i in range(len(data) - t - 1):
		a = data[i:(i+t)]
		dataX.append(a)
		dataY.append(data[i + t])
	return np.array(dataX), np.array(dataY)

def splitData(data, v):
    train_size = int(len(data[0]) * v) # v is between 0 and 1
    train, test = (data[0][0:train_size],data[1][0:train_size]), (data[0][train_size:len(data[0])],data[1][train_size:len(data[0])])
    return train, test

def sliceData(data, s):
    dataX, dataY = data[0], data[1]
    seq = randint(1,len(data[0])-20)
    sliceX = dataX[seq:seq+s]
    sliceY = dataY[seq:seq+s]
    return sliceX, sliceY

#%%
   
file = 'input.txt'
abc_list = loadMusic(file) # read in data
lvl = np.min(abc_list)
abc_list = abc_list - lvl
vocab = np.max(abc_list)



data = prepareData(abc_list, 1) # prepare inputs and targets
#dataslice = sliceData(data, s = 2) # slice into sequences of length s
train_seq, test_seq = splitData(data, 0.8)
y_train = train_seq[1]
y_test = test_seq[1]

# one-hot encoding targets
X_train = np_utils.to_categorical(train_seq[0], vocab+1)
X_test = np_utils.to_categorical(test_seq[0], vocab+1)
Y_train = np_utils.to_categorical(train_seq[1], vocab+1)
Y_test = np_utils.to_categorical(test_seq[1], vocab+1)
 
seq = 24; # sequence length
X_train = X_train[0:int(X_train.shape[0]/seq)*seq][:]
Y_train = Y_train[0:int(Y_train.shape[0]/seq)*seq][:]
X_test = X_test[0:int(X_test.shape[0]/seq)*seq][:]
Y_test = Y_test[0:int(Y_test.shape[0]/seq)*seq][:]


X_train = X_train.reshape(int(X_train.shape[0]/seq), seq, X_train.shape[1]) # reshape input for RNN
X_test = X_test.reshape(int(X_test.shape[0]/seq), seq, X_test.shape[1])
Y_train = Y_train.reshape(int(Y_train.shape[0]/seq), seq, Y_train.shape[1]) # reshape input for RNN
Y_test = Y_test.reshape(int(Y_test.shape[0]/seq), seq, Y_test.shape[1])


# define hyperparameters
hid_neurons = 100
nb_classes = np.shape(Y_train)[-1]


#%% create model
model = Sequential()
model.add(SimpleRNN(output_dim = hid_neurons,
                    init='glorot_uniform', inner_init='orthogonal', activation='relu',
                    W_regularizer=None, U_regularizer=None, b_regularizer=None,
                    dropout_W=0.0, dropout_U=0.0, return_sequences = True,
                    input_shape = X_train.shape[1:]))
model.add(Dense(output_dim = nb_classes))
model.add(Activation('softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
#%% train model and save weights
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
history = model.fit(X_train, Y_train, batch_size = 100, 
                       nb_epoch = 500, validation_data = (X_test, Y_test), shuffle = False)
#%%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
#plt.title('model loss for %d samples per category')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
#plt.title('model accuracy for %d samples per category' % samCat)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()
#%% load weights

filename = "weights-improvement-19-1.9435.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

#%% generating music

seed = X_train[1]
pattern = seed
for i in range(0,1000):
    	x = np.reshape(pattern, (1, pattern.shape[0], pattern.shape[1]))
    	prediction = model.predict(x, verbose=1)
     	index = np.argmax(prediction, axis = 2)
      




