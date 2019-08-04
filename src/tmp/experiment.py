#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import keras
import pickle
import os
import re
import pandas as pd
import numpy as np
from glob import glob
from train_rnn import load_data
from train_svm import loadData
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import classification_report

def runRNN(pickle_file,X_test_rnn,y_test):
    model = keras.models.load_model(glob("./pickles/"+pickle_file+"/best*")[0])
    if "words" in pickle_file:
        out = model.predict(X_test_rnn[0])
    elif "char" in pickle_file:
        out = model.predict(X_test_rnn[1])
    elif "all" in pickle_file:
        out = model.predict([X_test_rnn[0],X_test_rnn[1]])
    out = np.where(out >= 0.5, 1, 0)
    with open("./pickles/"+pickle_file+"/classification_report_valid.txt", "w") as f:
        f.write(classification_report(y_test,out,digits=4))
    
def runSVM(pickle_file,X_test_svm,y_test):
    full_name = glob("./pickles/"+pickle_file+"/best*")[0]
    with open(full_name,"rb") as f:
        model = pickle.load(f)
    if "rbf" in pickle_file:
        number = int(re.sub(".pickle","",re.sub(r".*best_model_","",full_name)))
        df = pd.read_csv(glob("./pickles/"+pickle_file+"/log*")[0])
        g = float(df[df["model"] == number]["gamma"])
        n = int(df[df["model"] == number]["n_components"])
        rbf_feature = RBFSampler(gamma=g,n_components=n)
        X_test_svm = rbf_feature.fit_transform(X_test_svm)
    out = model.predict(X_test_svm)
    with open("./pickles/"+pickle_file+"/classification_report_valid.txt", "w") as f:
        f.write(classification_report(y_test,out,digits=4))

_,X_valid_rnn, _,_,y_valid_rnn,_ = load_data("all")
_,_,X_valid_svm,y_valid_svm,_,_ = loadData()
files = glob("./pickles/20*")

for file in files:
    filename = os.path.basename(file)
    if "rnn" in file:
        runRNN(filename,X_valid_rnn,y_valid_rnn)
        # blindRNN(filename,blind_data,text,y_blind,args.padding_tokens,args.padding_char)
    elif "svm" in file:
        runSVM(filename,X_valid_svm,y_valid_svm)
        # blindSVM(filename,text,y_blind)