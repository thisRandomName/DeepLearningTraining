from __future__ import print_function

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM

from sklearn.model_selection import train_test_split
import statistics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import statistics
from numpy import array
from numpy import argmax
from keras.utils import to_categorical


max_features = 20000
# cut texts after this number of words (among top max_features most common words)
maxlen = 80
batch_size = 32



# load files
train = pd.read_csv('/kaggle/input/question3/train.csv')
# train=train.head(n=500)
test = pd.read_csv('/kaggle/input/question3/test_without_labels.csv')

# def format_data( for tokenization and pad_sequences)
def format_data(train, test, max_features, maxlen):
  
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.utils import to_categorical
    
    train = train.sample(frac=1).reset_index(drop=True)
    train['Content'] = train['Content'].apply(lambda x: x.lower())
    test['Content'] = test['Content'].apply(lambda x: x.lower())

    X = train['Content']
    test_X = test['Content']
    Y = to_categorical(train['Label'].values)

    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(X))

    X = tokenizer.texts_to_sequences(X)
    X = pad_sequences(X, maxlen=maxlen)
    test_X = tokenizer.texts_to_sequences(test_X)
    test_X = pad_sequences(test_X, maxlen=maxlen)

    return X, Y, test_X

# Here we split data to training and testing parts
X, Y, test_X = format_data(train, test, max_features, maxlen)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.25, random_state=13)


print('Building model')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(X_val, Y_val))

score, acc = model.evaluate(X_val, Y_val,
                            batch_size=batch_size)

# print('Test score:', score)
print('Test accuracy:', acc)


#make predictions

data = array(Y_val)

# data=data[:,0]
data=np.argmax(data, axis=1)
print(data)

ypred1 = model.predict(X_val)
ypred = np.argmax(ypred1, axis=1)
print(ypred)


# precision tp / (tp + fp)
precision = precision_score(data,ypred,average='macro')
print('Precision: %f' % precision)

# recall: tp / (tp + fn)
recall = recall_score(data,ypred,average='macro')
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(data,ypred,average='macro')
print('F1 score: %f' % f1)

