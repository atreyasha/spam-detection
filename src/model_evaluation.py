#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os
import keras
import pickle
import argparse
import pandas as pd
import numpy as np
from copy import deepcopy
from glob import glob
from train_rnn import load_data
from train_svm import loadData
from bag_words import tokenize, bagging
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report,roc_auc_score
from sklearn.kernel_approximation import RBFSampler
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence

#############################
# blind data processing
#############################

def readBlind():
    text = []
    labels = []
    with open("./data/blind/SMSSpamCollection","r") as f:
        for line in f:
            content = line.split()
            if content[0] == "ham":
                labels.append(0)
            elif content[0] == "spam":
                labels.append(1)
            text.append(content[1:])
    # process gathered text
    y_blind = np.asarray(labels)
    text = [" ".join(el) for el in text]
    blind_data = [text_to_word_sequence(el) for el in text]
    return text, blind_data, y_blind

def blindRNN(pickle_file,blind_data,text,y_blind,maxlen_words=500,maxlen_char=1000):
    if "words" in pickle_file or "all" in pickle_file:
        with open("./data/rnn/words/integer_index_tokens.pickle", "rb") as f:
            word_dict = pickle.load(f)
        X_blind_words = pad_sequences([[word_dict[el] if el in word_dict.keys() else 0 
                                  for el in ls] for ls in blind_data], maxlen=maxlen_words)
    if "char" in pickle_file or "all" in pickle_file:
        with open("./data/rnn/char/integer_index_char.pickle", "rb") as f:
            char_dict = pickle.load(f)
        X_blind_char = pad_sequences([[char_dict[el.lower()] if el.lower() in char_dict.keys() else 0 
                                              for el in ls] for ls in text], maxlen=maxlen_char)
    model = keras.models.load_model(glob("./pickles/"+pickle_file+"/best*")[0])
    if "words" in pickle_file:
        out = model.predict(X_blind_words)
    elif "char" in pickle_file:
        out = model.predict(X_blind_char)
    elif "all" in pickle_file:
        out = model.predict([X_blind_words,X_blind_char])
    roc = roc_auc_score(y_blind,out)
    out = np.where(out >= 0.5, 1, 0)
    with open("./pickles/"+pickle_file+"/classification_report_blind.txt", "w") as f:
        f.write("ROC: "+str(roc)+"\n")
        f.write(classification_report(y_blind,out,digits=4))

def blindSVM(pickle_file,text,y_blind):
    y_svm_blind = deepcopy(y_blind)
    cleaned = tokenize(text)
    with open("./data/svm/words/integer_index_tokens.pickle","rb") as f:
        word_dict = pickle.load(f)
    X_blind_words = bagging(cleaned,word_dict)
    full_name = glob("./pickles/"+pickle_file+"/best*")[0]
    with open(full_name,"rb") as f:
        model = pickle.load(f)
    y_svm_blind[np.where(y_svm_blind==0)[0]] = -1
    if "rbf" in pickle_file:
        number = int(re.sub(".pickle","",re.sub(r".*best_model_","",full_name)))
        df = pd.read_csv(glob("./pickles/"+pickle_file+"/log*")[0])
        g = float(df[df["model"] == number]["gamma"])
        n = int(df[df["model"] == number]["n_components"])
        rbf_feature = RBFSampler(gamma=g,n_components=n)
        X_blind_words = rbf_feature.fit_transform(X_blind_words)
    out = model.decision_function(X_blind_words)
    roc = roc_auc_score(y_svm_blind,out)
    with open("./pickles/"+pickle_file+"/classification_report_blind.txt", "w") as f:
        f.write("ROC: "+str(roc)+"\n")
        f.write(classification_report(y_svm_blind,model.predict(X_blind_words),digits=4))

#############################
# save prob. maps for models
#############################

def runRNN(pickle_file,X_test_rnn):
    model = keras.models.load_model(glob("./pickles/"+pickle_file+"/best*")[0])
    if "words" in pickle_file:
        out = model.predict(X_test_rnn[0])
    elif "char" in pickle_file:
        out = model.predict(X_test_rnn[1])
    elif "all" in pickle_file:
        out = model.predict([X_test_rnn[0],X_test_rnn[1]])
    np.save("./pickles/"+pickle_file+"/prob_map_test.npy", out)
    
def runSVM(pickle_file,X_test_svm):
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
    out = model.decision_function(X_test_svm)
    np.save("./pickles/"+pickle_file+"/prob_map_test.npy", out)
    
##############################
# main command call
##############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--padding-tokens", type=int, default = 500,
                        help="maximum length of email padding for tokens <default:500>")
    parser.add_argument("--padding-char", type=int, default = 1000,
                        help="maximum length of email padding for characters <default:1000>")
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-p', '--pickle', type=str,
                               help="pickle directory name for stored model, or input 'all' to run on all models", 
                               required=True)
    args = parser.parse_args()
    # execute main command
    text, blind_data, y_blind = readBlind()
    _,_, X_test_rnn,_,_,_ = load_data("all")
    _,_, _, _,X_test_svm,_ = loadData()
    files = glob("./pickles/20*")
    # run evaluations based on pickle input
    if args.pickle != "all":
        if "rnn" in args.pickle:
            runRNN(args.pickle,X_test_rnn)
            blindRNN(args.pickle,blind_data,text,y_blind,args.padding_tokens,args.padding_char)
        elif "svm" in args.pickle:
            runSVM(args.pickle,X_test_svm)
            blindSVM(args.pickle,text,y_blind)
    else:
        for file in files:
            filename = os.path.basename(file)
            if "rnn" in file:
                runRNN(filename,X_test_rnn)
                blindRNN(filename,blind_data,text,y_blind,args.padding_tokens,args.padding_char)
            elif "svm" in file:
                runSVM(filename,X_test_svm)
                blindSVM(filename,text,y_blind)