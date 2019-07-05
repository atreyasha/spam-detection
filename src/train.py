#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################
# LSTM + conv1D
##############################

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.backend.tensorflow_backend import clear_session

##############################
# save np.load
np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True,**k)
# call load_data with allow_pickle implicitly set to true
# X_train = np.load("./data/X_train_body.npy")
# X_valid = np.load("./data/X_valid_body.npy")
# X_test = np.load("./data/X_test_body.npy")
X_train = np.load("./data/X_train_subject.npy")
X_valid = np.load("./data/X_valid_subject.npy")
X_test = np.load("./data/X_test_subject.npy")
y_train = np.load("./data/y_train.npy")
y_valid = np.load("./data/y_valid.npy")
y_test = np.load("./data/y_test.npy")
# restore np.load for future normal usage
np.load = np_load_old
##############################
# truncate and pad input sequences
# this would be vocab_size + 1
vocab_size = 5001
# max_review_length = 500
max_review_length = 50
embedding_vector_length = 300
# create the model
clear_session()
model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_length, input_length=max_review_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=3, batch_size=64, validation_data=(X_valid, y_valid), shuffle=True)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))
# 88% accuracy on test set

##############################
# comments/todos
##############################

# architecture:
# TODO: run pipeline with LSTM for body and subject
# set baseline with body text only
# initialize with glove embeddings and train from there
# improve text from then on, and take note of improvements that come along
# compare with bag-of-words with SVM
# add regularization with dropout
# split data into test, validation and training
# use validation for early stopping and checks
# make two sets of word vectors, for body and subject
# use NER tagger to remove named entities for more generality
# think of unknown word handling, maybe skip or add unknown vector or character embedding
# uniform classifier will already have 50% accuracy, or can be exactly calculated
# once model is satisfactory, can allow for hypertuning measures

# data encoding:
# TODO: maybe separate embedding for email subjects, in that case maybe separate encoding too 
# set threshold to stop vocabulary, example where the frequency should stop
# have integer label for named entities, do not use them in integer labeling
# find out distrbution of tokens in both

# extra:
# TODO: try attention-based mechanism
# maybe add language as additional input
# consider character vectors for unknown words
# only use spam filter if unknown address or not yourself
# see if spam model can actually re-generate spam text, which would be interesting
# make two separate models, then combine them to see if there is a significant effect