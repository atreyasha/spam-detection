#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################
# import dependencies
##############################

import re
import pickle
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split 

##############################
# define key functions
##############################

def read_order(files):
    # read in files
    corpus = []
    for i in range(len(files)):
        with open(files[i], "r", encoding="latin-1") as reader:
            corpus.append(reader.readlines())
    # order them well
    for i, el in enumerate(corpus):
        if type(el) == list:
            for j, line in enumerate(corpus[i]):
                if "subject" in corpus[i][j] or "Subject" in corpus[i][j]:
                    corpus[i] = [re.sub("subject: ","",corpus[i][j].strip(), flags=re.IGNORECASE), [" ".join([rem.strip() for rem in corpus[i][:j]+corpus[i][j+1:]])]]
                    break
    return corpus

def tokenize(obj):
    return [[text_to_word_sequence(el[0]),text_to_word_sequence(el[1][0])] for el in obj]

def integerEncode(vocab_size=5000, padding_body = 500, padding_subject = 50):
    ham = [filename for filename in Path('./data/enron').glob('**/*ham.txt')]
    spam = [filename for filename in Path('./data/enron').glob('**/*spam.txt')]
    hamCorpus = read_order(ham)
    spamCorpus = read_order(spam)
    label_ham = [0 for _ in range(len(hamCorpus))]
    label_spam = [1 for _ in range(len(spamCorpus))]
    # separate into training, test
    ham_train, ham_test, ham_train_labels, ham_test_labels = train_test_split(hamCorpus,label_ham,test_size=0.33,random_state=42)
    spam_train, spam_test, spam_train_labels, spam_test_labels = train_test_split(spamCorpus,label_spam,test_size=0.33,random_state=42)
    # re-separate into training and validation
    ham_train, ham_valid, ham_train_labels, ham_valid_labels = train_test_split(ham_train,ham_train_labels,test_size=0.15,random_state=42)
    spam_train, spam_valid, spam_train_labels, spam_valid_labels = train_test_split(spam_train,spam_train_labels,test_size=0.15,random_state=42)
    # create train dataset
    X_train = tokenize(ham_train + spam_train)
    y_train = np.array(ham_train_labels + spam_train_labels)
    # create validation dataset
    X_valid = tokenize(ham_valid + spam_valid)
    y_valid = np.array(ham_valid_labels + spam_valid_labels)
    # create test dataset
    X_test = tokenize(ham_test + spam_test)
    y_test = np.array(ham_test_labels + spam_test_labels)
    # integer encode data
    unknown_token = vocab_size
    tokens = [el[0]+el[1] for el in X_train]
    tokens_flat = [token for el in tokens for token in el]
    tokens_common = Counter(tokens_flat).most_common()
    tokens_common = dict(tokens_common[:unknown_token-1])
    for i, key in enumerate(tokens_common.keys()):
        tokens_common[key] = i+1
    # encode training set
    X_train_subject = pad_sequences([[tokens_common[el] if el in tokens_common.keys() else unknown_token 
                                      for el in ls[0]] for ls in X_train], maxlen=padding_subject)
    X_train_body = pad_sequences([[tokens_common[el] if el in tokens_common.keys() else unknown_token 
                                   for el in ls[1]] for ls in X_train], maxlen=padding_body)
    # encode validation set
    X_valid_subject = pad_sequences([[tokens_common[el] if el in tokens_common.keys() else unknown_token 
                                      for el in ls[0]] for ls in X_valid],maxlen=padding_subject)
    X_valid_body = pad_sequences([[tokens_common[el] if el in tokens_common.keys() else unknown_token 
                                   for el in ls[1]] for ls in X_valid],maxlen=padding_body)
    # encode test set
    X_test_subject = pad_sequences([[tokens_common[el] if el in tokens_common.keys() else unknown_token 
                                     for el in ls[0]] for ls in X_test],maxlen=padding_subject)
    X_test_body = pad_sequences([[tokens_common[el] if el in tokens_common.keys() else unknown_token 
                                  for el in ls[1]] for ls in X_test], maxlen=padding_body)
    # save all numpy arrays to file
    np.save("./data/X_train_subject.npy", X_train_subject)
    np.save("./data/X_valid_subject.npy", X_valid_subject)
    np.save("./data/X_test_subject.npy", X_test_subject)
    np.save("./data/X_train_body.npy", X_train_body)
    np.save("./data/X_valid_body.npy", X_valid_body)
    np.save("./data/X_test_body.npy", X_test_body)
    np.save("./data/y_train.npy", y_train)
    np.save("./data/y_valid.npy", y_valid)
    np.save("./data/y_test.npy", y_test)
    # save indexing dictionary to file
    with open("./data/integer_index.pickle", "wb") as f:
        pickle.dump(tokens_common, f, protocol=pickle.HIGHEST_PROTOCOL)

##############################
# main command call
##############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default = 5000,
                            help="size of vocabulary used in word vector embedding <default:5000>")
    parser.add_argument("--padding-body", type=int, default = 500,
                        help="maximum length of email body padding <default:500>")
    parser.add_argument("--padding-subject", type=int, default = 50,
                        help="maximum length of email subject padding <default:50>")
    args = parser.parse_args()
    # execute main command
    integerEncode(vocab_size=args.vocab_size, padding_body=args.padding_body, padding_subject=args.padding_subject)