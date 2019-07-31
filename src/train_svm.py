#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import argparse
import datetime
import pickle
import numpy as np
from glob import glob
from copy import deepcopy
from tqdm import tqdm
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import accuracy_score

##############################
# define key functions
##############################

def getCurrentTime():
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

def batchDispatcher(X,batch_size):
    no_samples = int(np.ceil(X.shape[0]/batch_size))
    return [np.random.randint(X.shape[0],size=batch_size) for _ in range(no_samples)]

def checkingTrainer(model,X_train,y_train,X_valid,y_valid,epochs,batch_size,patience):
    patience_counter = 0
    for epoch in range(epochs):
        print("\n Epoch: %s" % str(epoch+1))
        # define batches which are shuffled at each epoch
        batches = batchDispatcher(X_train,batch_size)
        for batch in tqdm(batches):
            # train via mini-batch SGD
            spec_y = y_train[batch]
            model.partial_fit(X_train[batch],spec_y,np.unique(spec_y))
        # predict on validation set at the end of each epoch
        val_acc = accuracy_score(y_valid,model.predict(X_valid))
        if epoch == 0:
            best_val = val_acc
            best_model = deepcopy(model)
        elif val_acc > best_val:
            # reset patience counter
            patience_counter = 0
            best_val = val_acc
            best_model = deepcopy(model)
        else:
            # check counter to stop training
            if patience_counter < patience:
                patience_counter += 1
            elif patience_counter == patience:
                break
    return best_val, best_model

def gridSearch(epochs=50,patience=5):
    # load labels
    y_train = np.load("./data/svm/y_train.npy")
    y_valid = np.load("./data/svm/y_valid.npy")
    y_test = np.load("./data/svm/y_test.npy")
    # load feature data
    X_train = np.load("./data/svm/words/X_train.npy")
    X_valid = np.load("./data/svm/words/X_valid.npy")
    X_test = np.load("./data/svm/words/X_test.npy")
    # normalize X training data
    X_train = X_train*(1/(np.sum(X_train, axis = 1)[:,None]))
    X_valid = X_valid*(1/(np.sum(X_valid, axis = 1)[:,None]))
    X_test = X_test*(1/(np.sum(X_test, axis = 1)[:,None]))
    # write log file and csv
    current_time = getCurrentTime()+"_svm"
    os.makedirs("pickles/"+current_time)
    csvfile = open('pickles/'+ current_time + '/' + 'log.csv', 'w')
    fieldnames = ["model", "kernel", "alpha", "batch_size", "gamma", "n_components", "best_train", "best_val", "best_test"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    csvfile.flush()
    # define grid-search parameters
    alpha = np.linspace(0.0001,0.001,20)
    Batch_size = np.linspace(100,500,5,dtype=int)
    gamma = np.linspace(0.01,1,20)
    n_components = np.linspace(100,1000,4,dtype=int)
    # start loop for linear kernel
    counter = 0
    record_test = 0
    for batch_size in Batch_size:
        for a in alpha:
            model = SGDClassifier(alpha=a,n_jobs=-1)
            best_val, best_model = checkingTrainer(model,X_train,y_train,X_valid,y_valid,
                                                   epochs,batch_size,patience)
            best_train = accuracy_score(y_train,model.predict(X_train)) 
            best_test = accuracy_score(y_test,model.predict(X_test))
            # write to csv file in loop
            writer.writerow({"model":str(counter), "kernel":"linear", "alpha":str(a),
                             "batch_size":str(batch_size), "gamma":"None", "n_components":"None",
                             "best_train":str(best_train), "best_val":str(best_val), 
                             "best_test":str(best_test)})
            csvfile.flush()
            if best_test >= record_test:
                record_test = best_test
                with open("./pickles/"+current_time+"/best_model_"+str(counter)+".pickle", "wb") as f:
                    pickle.dump(best_model,f,protocol=pickle.HIGHEST_PROTOCOL)
                todel= [el for el in glob("./pickles/"+current_time+"/best_model*") if 'best_model_'+str(counter)+'.pickle' not in el]
                if len(todel) > 0:
                    for el in todel:
                        os.remove(el)
            counter += 1
    # start loop for rbf kernel
    for batch_size in Batch_size:
        for a in alpha:
            for g in gamma:
                for n in n_components:
                    feature_map_nystroem = Nystroem(gamma=g,n_components=n)
                    X_train_mod = feature_map_nystroem.fit_transform(X_train)
                    X_valid_mod = feature_map_nystroem.fit_transform(X_valid)
                    X_test_mod = feature_map_nystroem.fit_transform(X_test)
                    model = SGDClassifier(alpha=a,n_jobs=-1)
                    best_val, best_model = checkingTrainer(model,X_train_mod,y_train,X_valid_mod,y_valid,
                                                           epochs,batch_size,patience)
                    best_train = accuracy_score(y_train,model.predict(X_train_mod))
                    best_test = accuracy_score(y_test,model.predict(X_test_mod))
                    # write to csv file in loop
                    writer.writerow({"model":str(counter), "kernel":"rbf", "alpha":str(a),
                                     "batch_size":str(batch_size), "gamma":str(g), "n_components":str(n),
                                     "best_train":str(best_train), "best_val":str(best_val), 
                                     "best_test":str(best_test)})
                    csvfile.flush()
                    if best_test >= record_test:
                        record_test = best_test
                        with open("./pickles/"+current_time+"/best_model_"+str(counter), "wb") as f:
                            pickle.dump(best_model, f, protocol=pickle.HIGHEST_PROTOCOL)
                        todel= [el for el in glob("./pickles/"+current_time+"/best_model*") if 'best_model_'+str(counter)+'.pickle' not in el]
                        if len(todel) > 0:
                            for el in todel:
                                os.remove(el)
                    counter += 1
    csvfile.close()
    
###############################
# main command call
###############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50,
                        help="maximum number of epochs for training <default:50>")
    parser.add_argument("--patience", type=int, default=5,
                        help="patience for early stopping <default:5>")
    args = parser.parse_args()
    gridSearch(args.epochs,args.patience)

##############################
# comments/to-dos
##############################

# TODO: look through RNN training loop to see if there could be memory leak, esp in handling best test accuracies
# deploy all code to google colab and provide links
# modify all code to output precision and recall as well, maybe can repeat multiple times for best model