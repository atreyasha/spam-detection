#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################
# import dependencies
##############################

import os
import csv
import argparse
import datetime
import numpy as np
from glob import glob
from keras import backend
from keras import layers, models
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout, CuDNNLSTM
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.backend.tensorflow_backend import clear_session
from keras.callbacks import EarlyStopping, ModelCheckpoint

##############################
# define functions
##############################

def load_data(subtype="words"):
    # save np.load
    np_load_old = np.load
    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True,**k)
    # call load_data with allow_pickle implicitly set to true
    if subtype == "words":
        X_train = np.load("./data/words/X_train.npy")
        X_valid = np.load("./data/words/X_valid.npy")
        X_test = np.load("./data/words/X_test.npy")
    elif subtype == "char":
        X_train = np.load("./data/char/X_train.npy")
        X_valid = np.load("./data/char/X_valid.npy")
        X_test = np.load("./data/char/X_test.npy")
    elif subtype == "all":
        X_train = (np.load("./data/words/X_train.npy"),np.load("./data/char/X_train.npy"))
        X_valid = (np.load("./data/words/X_valid.npy"),np.load("./data/char/X_valid.npy"))
        X_test = (np.load("./data/words/X_test.npy"),np.load("./data/char/X_test.npy"))
    y_train = np.load("./data/y_train.npy")
    y_valid = np.load("./data/y_valid.npy")
    y_test = np.load("./data/y_test.npy")
    # restore np.load for future normal usage
    np.load = np_load_old
    return X_train, X_valid, X_test, y_train, y_valid, y_test
    
def getCurrentTime():
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
def getModel(subtype = "words", embedding_vector_size = 100, droprate = 0.2):
    if subtype == "words":
        vocab_size = 5001
        max_text_length = 500
    elif subtype == "char":
        vocab_size = 66
        max_text_length = 1000
    else:
        raise NameError("no such option provided: "+str(subtype))
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_vector_size, input_length=max_text_length))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same'))
    model.add(Conv1D(filters=64, kernel_size=3, padding='same'))
    model.add(Activation("relu"))
    model.add(Dropout(droprate))
    model.add(MaxPooling1D(pool_size=2))
    if len(backend.tensorflow_backend._get_available_gpus()) > 0:
        model.add(CuDNNLSTM(100))
        model.add(Dropout(droprate))
    else:
        model.add(LSTM(100, recurrent_dropout=droprate))
    model.add(Dense(50))
    model.add(Activation("relu"))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    return model

def singleRun(subtype="words"):
    if subtype in ["words", "char"]:
        X_train, X_valid, X_test, y_train, y_valid, y_test = load_data(subtype)
    model = getModel(subtype)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=3, batch_size=256,
                        validation_data=(X_valid, y_valid), shuffle=True)
    scores = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy: " + str(scores[1]*100) + "%")
    return model

def gridSearch(subtype="words"):
    # load data into memory
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_data(subtype)
    # create log directory and info csv
    current_time = getCurrentTime()+"_"+subtype
    os.makedirs("pickles/"+current_time)
    csvfile = open('pickles/'+ current_time + '/' + 'log.csv', 'w')
    fieldnames = ["model", "embedding_size", "droprate", "batch_size", "learning_rate", "best_train", "best_val", "best_test"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    csvfile.flush()
    # define grid parameters
    embedding_size = [50,100,300]
    droprate = np.linspace(0.1,0.4,4)
    batch_size = [128,256,384]
    learning_rate = np.linspace(0.001,0.01,4)
    counter = 0
    best = []
    best_test = []
    # run grid-search
    for e in embedding_size:
        for d in droprate:
            for b in batch_size:
                for l in learning_rate:
                    clear_session()
                    model = getModel(subtype,embedding_vector_size=e,droprate=d)
                    callbacks = [EarlyStopping(monitor='val_acc', patience=5, restore_best_weights=True),
                                 ModelCheckpoint(filepath='./pickles/'+current_time+'/best_model_'
                                                 +str(counter)+'.h5', monitor='val_acc', save_best_only=True)]
                    model.compile(optimizer=Adam(lr=l), loss="binary_crossentropy", metrics=['accuracy'])
                    history = model.fit(X_train, y_train, epochs=50, batch_size=b, 
                                             validation_data=(X_valid, y_valid), shuffle=True,
                                             callbacks=callbacks)
                    max_index = np.argmax(history.history["val_acc"])
                    scores = model.evaluate(X_test, y_test, verbose=1)
                    best_test.append(scores[1])
                    best.append([history.history["acc"][max_index],history.history["val_acc"][max_index],scores[1]])
                    if np.argmax(best_test) == counter:
                        todel= [el for el in glob("./pickles/"+current_time+"/best_model*") if 'best_model_'+str(counter)+'.h5' not in el]
                        if len(todel) > 0:
                            for el in todel:
                                os.remove(el)
                    else:
                        os.remove('./pickles/'+current_time+'/best_model_'+str(counter)+'.h5')
                    # write to csv file in loop
                    writer.writerow({"model":str(counter), "embedding_size":str(e), "droprate":str(d), 
                                     "batch_size":str(b), "learning_rate":str(l),
                                     "best_train":str(best[counter][0]), "best_val":str(best[counter][1]), 
                                     "best_test":str(best[counter][2])})
                    csvfile.flush()
                    counter += 1
                    # clear memory
                    del model
                    del callbacks
                    del history
    csvfile.close()
    return 0
    
def plot_K_model(name,subtype="words"):
    # code adopted from https://github.com/keras-team/keras/issues/10386
    # useful to convert from sequential to functional for plotting model
    clear_session()
    model = getModel(subtype)
    input_layer = layers.Input(batch_shape=model.layers[0].input_shape)
    prev_layer = input_layer
    for layer in model.layers:
        prev_layer = layer(prev_layer)
    funcmodel = models.Model([input_layer], [prev_layer])
    plot_model(funcmodel, to_file='../img/'+name+'.png', show_shapes=True)
    
###############################
# main command call
###############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtype", type=str, default="words",
                        help="which model subtype to use; either 'words', 'char' or 'all' <default:'words'>")
    parser.add_argument("--grid-search", default=True, action="store_true",
                        help="option to conduct grid-search, enabled by default")
    parser.add_argument("--single-run", default=False, action="store_true",
                        help="option to conduct single run based on default hyperparameters, disabled by default")
    parser.add_argument("--plot", default=False, action="store_true", 
                        help="option for plotting keras model, disabled by default")
    parser.add_argument("--name", type=str, default="model",
            help="if --plot option is chosen, this provides name of the model image <default:'model'>")
    args = parser.parse_args()
    assert args.subtype in ["words","char","all"]
    if args.plot:
        plot_K_model(args.name,args.subtype)
    elif args.single_run:
        singleRun(args.subtype)
    elif args.grid_search:
        gridSearch(args.subtype)