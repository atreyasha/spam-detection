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
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Activation, concatenate
from keras.layers import LSTM, Dropout, CuDNNLSTM, Input
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
        X_train = np.load("./data/rnn/words/X_train.npy")
        X_valid = np.load("./data/rnn/words/X_valid.npy")
        X_test = np.load("./data/rnn/words/X_test.npy")
    elif subtype == "char":
        X_train = np.load("./data/rnn/char/X_train.npy")
        X_valid = np.load("./data/rnn/char/X_valid.npy")
        X_test = np.load("./data/rnn/char/X_test.npy")
    elif subtype == "all":
        X_train = (np.load("./data/rnn/words/X_train.npy"),np.load("./data/rnn/char/X_train.npy"))
        X_valid = (np.load("./data/rnn/words/X_valid.npy"),np.load("./data/rnn/char/X_valid.npy"))
        X_test = (np.load("./data/rnn/words/X_test.npy"),np.load("./data/rnn/char/X_test.npy"))
    y_train = np.load("./data/rnn/y_train.npy")
    y_valid = np.load("./data/rnn/y_valid.npy")
    y_test = np.load("./data/rnn/y_test.npy")
    # restore np.load for future normal usage
    np.load = np_load_old
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def getCurrentTime():
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
def getModel(embedding_matrix_words,embedding_matrix_char,
             subtype = "words", embedding_vector_size = 300, droprate = 0.2, vocab_size_tokens = 5000,
             max_text_length_tokens = 500, vocab_size_char = 65, max_text_length_char = 1000):
    if subtype in ["words","char"]:
        model = Sequential()
        if subtype == "words":
            if embedding_matrix_words is not None:
                model.add(Embedding(vocab_size_tokens, embedding_vector_size,
                                input_length=max_text_length_tokens, weights=[embedding_matrix_words]))
            else:
                model.add(Embedding(vocab_size_tokens, embedding_vector_size,
                                input_length=max_text_length_tokens))
        elif subtype == "char":
            if embedding_matrix_words is not None:
                model.add(Embedding(vocab_size_char, embedding_vector_size, input_length=max_text_length_char,
                                weights=[embedding_matrix_char]))
            else:
                model.add(Embedding(vocab_size_char, embedding_vector_size, input_length=max_text_length_char))
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
    elif subtype == "all":
        # define token-based model
        input_tokens = Input(shape=(max_text_length_tokens,))
        if embedding_matrix_words is not None:
            tokens = Embedding(vocab_size_tokens, embedding_vector_size, 
                           input_length=max_text_length_tokens, weights=[embedding_matrix_words])(input_tokens)
        else:
            tokens = Embedding(vocab_size_tokens, embedding_vector_size, 
                           input_length=max_text_length_tokens)(input_tokens)
        tokens = Conv1D(filters=32, kernel_size=3, padding='same')(tokens)
        tokens = Conv1D(filters=64, kernel_size=3, padding='same')(tokens)
        tokens = Activation("relu")(tokens)
        tokens = Dropout(droprate)(tokens)
        tokens = MaxPooling1D(pool_size=2)(tokens)
        if len(backend.tensorflow_backend._get_available_gpus()) > 0:
            tokens = CuDNNLSTM(100)(tokens)
            tokens = Dropout(droprate)(tokens)
        else:
            tokens = LSTM(100, recurrent_dropout=droprate)(tokens)
        tokens = Model(inputs=input_tokens,outputs=tokens)
        # define-character based model
        input_char = Input(shape=(max_text_length_char,))
        if embedding_matrix_char is not None:
            char = Embedding(vocab_size_char, embedding_vector_size, 
                         input_length=max_text_length_char, weights=[embedding_matrix_char])(input_char)
        else: 
            char = Embedding(vocab_size_char, embedding_vector_size, 
                         input_length=max_text_length_char)(input_char)
        char = Conv1D(filters=32, kernel_size=3, padding='same')(char)
        char = Conv1D(filters=64, kernel_size=3, padding='same')(char)
        char = Activation("relu")(char)
        char = Dropout(droprate)(char)
        char = MaxPooling1D(pool_size=2)(char)
        if len(backend.tensorflow_backend._get_available_gpus()) > 0:
            char = CuDNNLSTM(100)(char)
            char = Dropout(droprate)(char)
        else:
            char = LSTM(100, recurrent_dropout=droprate)(char)
        char = Model(inputs=input_char,outputs=char)
        combined = concatenate([tokens.output, char.output])
        combined = Dense(50)(combined)
        combined = Activation("relu")(combined)
        combined = Dense(1)(combined)
        combined = Activation("sigmoid")(combined)
        model = Model(inputs=[tokens.input, char.input], outputs=combined)
        return model

def singleRun(subtype="words",pre_trained_embeddings=True):
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_data(subtype)
    if subtype == "words":
        if pre_trained_embeddings:
            embedding_matrix_words = np.load("./data/glove/glove.6B.300d_word_emb.npy")
        else:
            embedding_matrix_words = None
        embedding_matrix_char = None
    elif subtype == "char":
        if pre_trained_embeddings:
            embedding_matrix_char = np.load("./data/glove/glove.6B.300d_char_emb.npy")
        else:
            embedding_matrix_char = None
        embedding_matrix_words = None
    else:
        if pre_trained_embeddings:
            embedding_matrix_words = np.load("./data/glove/glove.6B.300d_word_emb.npy")
            embedding_matrix_char = np.load("./data/glove/glove.6B.300d_char_emb.npy")
        else:
             embedding_matrix_words = None
             embedding_matrix_char = None
    model = getModel(embedding_matrix_words,embedding_matrix_char,subtype)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
    if subtype in ["words","char"]:
        model.fit(X_train, y_train, epochs=3, batch_size=256,
                            validation_data=(X_valid, y_valid), shuffle=True)
        scores = model.evaluate(X_test, y_test, verbose=1)
        print("Accuracy: " + str(scores[1]*100) + "%")
    elif subtype == "all":
        model.fit([X_train[0],X_train[1]], y_train, epochs=3, batch_size=256,
                            validation_data=([X_valid[0],X_valid[1]], y_valid), shuffle=True)
        scores = model.evaluate([X_test[0],X_test[1]], y_test, verbose=1)
        print("Accuracy: " + str(scores[1]*100) + "%")
    return model

def gridSearch(subtype="words",pre_trained_embeddings=True):
    # load data into memory
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_data(subtype)
    # create log directory and info csv
    if pre_trained_embeddings:
        current_time = getCurrentTime()+"_rnn_"+subtype+"_glove_embed"
    else:
        current_time = getCurrentTime()+"_rnn_"+subtype+"_random_embed"
    os.makedirs("pickles/"+current_time)
    csvfile = open('pickles/'+ current_time + '/' + 'log.csv', 'w')
    fieldnames = ["model", "pre_trained_embeddings", "embedding_size", "droprate", "batch_size", "learning_rate", "best_train", "best_val", "best_test"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    csvfile.flush()
    # define grid parameters
    embedding_size = [50,100,200,300]
    droprate = np.linspace(0.1,0.3,3)
    batch_size = [128,256]
    learning_rate = np.linspace(0.001,0.006,3)
    counter = 0
    record_test = 0
    # run grid-search
    for e in embedding_size:
        # save np.load
        np_load_old = np.load
        # modify the default parameters of np.load
        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True,**k)
        if subtype == "words":
            if pre_trained_embeddings:
                embedding_matrix_words = np.load("./data/glove/glove.6B."+str(e)+"d_word_emb.npy")
            else:
                embedding_matrix_words = None
            embedding_matrix_char = None
        elif subtype == "char":
            if pre_trained_embeddings:
                embedding_matrix_char = np.load("./data/glove/glove.6B."+str(e)+"d_char_emb.npy")
            else:
                embedding_matrix_char = None
            embedding_matrix_words = None
        else:
            if pre_trained_embeddings:
                embedding_matrix_words = np.load("./data/glove/glove.6B."+str(e)+"d_word_emb.npy")
                embedding_matrix_char = np.load("./data/glove/glove.6B."+str(e)+"d_char_emb.npy")
            else:
                 embedding_matrix_words = None
                 embedding_matrix_char = None
        np.load = np_load_old
        # move into grid-search loop
        for d in droprate:
            for b in batch_size:
                for l in learning_rate:
                    clear_session()
                    model = getModel(embedding_matrix_words,embedding_matrix_char,
                                     subtype,embedding_vector_size=e,droprate=d)
                    callbacks = [EarlyStopping(monitor='val_acc', patience=5, restore_best_weights=True),
                                 ModelCheckpoint(filepath='./pickles/'+current_time+'/best_model_'
                                                 +str(counter)+'.h5', monitor='val_acc', save_best_only=True)]
                    model.compile(optimizer=Adam(lr=l), loss="binary_crossentropy", metrics=['accuracy'])
                    if subtype in ["words","char"]:
                        history = model.fit(X_train, y_train, epochs=50, batch_size=b,
                                            validation_data=(X_valid, y_valid), shuffle=True,
                                            callbacks=callbacks)
                        scores = model.evaluate(X_test, y_test, verbose=1)
                        print("Accuracy: " + str(scores[1]*100) + "%")
                    elif subtype == "all":
                        history = model.fit([X_train[0],X_train[1]], y_train, epochs=50, batch_size=b,
                                            validation_data=([X_valid[0],X_valid[1]], y_valid), shuffle=True,
                                            callbacks=callbacks)
                        scores = model.evaluate([X_test[0],X_test[1]], y_test, verbose=1)
                        print("Accuracy: " + str(scores[1]*100) + "%")
                    max_index = np.argmax(history.history["val_acc"])
                    best_test = scores[1]
                    if best_test >= record_test:
                        record_test = best_test
                        todel= [el for el in glob("./pickles/"+current_time+"/best_model*") if 'best_model_'+str(counter)+'.h5' not in el]
                        if len(todel) > 0:
                            for el in todel:
                                os.remove(el)
                    else:
                        os.remove('./pickles/'+current_time+'/best_model_'+str(counter)+'.h5')
                    # write to csv file in loop
                    writer.writerow({"model":str(counter), "pre_trained_embeddings":str(pre_trained_embeddings),
                                     "embedding_size":str(e), "droprate":str(d),
                                     "batch_size":str(b), "learning_rate":str(l),
                                     "best_train":str(history.history["acc"][max_index]),
                                     "best_val":str(history.history["val_acc"][max_index]),
                                     "best_test":str(best_test)})
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
    if subtype in ["words","char"]:
        clear_session()
        model = getModel(None,None,subtype)
        input_layer = layers.Input(batch_shape=model.layers[0].input_shape)
        prev_layer = input_layer
        for layer in model.layers:
            prev_layer = layer(prev_layer)
        funcmodel = models.Model([input_layer], [prev_layer])
    elif subtype == "all":
        funcmodel = getModel(None,None,subtype)
    plot_model(funcmodel, to_file='../img/'+name+'.png', show_shapes=True)
    
###############################
# main command call
###############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtype", type=str, default="words",
                        help="which model subtype to use; either 'words', 'char' or 'all' <default:'words'>")
    parser.add_argument("--pre-trained-embeddings", default=False, action="store_true",
                        help="option to use pre-trained word/character embeddings, disabled by default")
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
        singleRun(args.subtype,args.pre_trained_embeddings)
    elif args.grid_search:
        gridSearch(args.subtype,args.pre_trained_embeddings)