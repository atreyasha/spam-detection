#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################
# import dependencies
##############################

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
    corpus = [[el.strip() for el in ls] for ls in corpus]
    corpus = [" ".join(ls) for ls in corpus]
    return corpus

def tokenize(obj):
    return [text_to_word_sequence(el) for el in obj]

def integerEncode(vocab_size=5000, padding_words=500, padding_chars=1000):
    ham = [filename for filename in Path('./data/enron').glob('**/*ham.txt')]
    spam = [filename for filename in Path('./data/enron').glob('**/*spam.txt')]
    hamCorpus = read_order(ham)
    spamCorpus = read_order(spam)
    label_ham = [0 for _ in range(len(hamCorpus))]
    label_spam = [1 for _ in range(len(spamCorpus))]
    # separate into training, test
    ham_train, ham_test, ham_train_labels, ham_test_labels = train_test_split(hamCorpus,
                                                                              label_ham,test_size=0.33,random_state=42)
    spam_train, spam_test, spam_train_labels, spam_test_labels = train_test_split(spamCorpus,
                                                                                  label_spam,test_size=0.33,random_state=42)
    # re-separate into training and validation
    ham_train, ham_valid, ham_train_labels, ham_valid_labels = train_test_split(ham_train,
                                                                                ham_train_labels,test_size=0.15,random_state=42)
    spam_train, spam_valid, spam_train_labels, spam_valid_labels = train_test_split(spam_train,
                                                                                    spam_train_labels,test_size=0.15,random_state=42)
    # create train dataset
    X_train = tokenize(ham_train + spam_train)
    y_train = np.array(ham_train_labels + spam_train_labels)
    # create validation dataset
    X_valid = tokenize(ham_valid + spam_valid)
    y_valid = np.array(ham_valid_labels + spam_valid_labels)
    # create test dataset
    X_test = tokenize(ham_test + spam_test)
    y_test = np.array(ham_test_labels + spam_test_labels)
    # save labels which will stay fixed
    np.save("./data/y_train.npy", y_train)
    np.save("./data/y_valid.npy", y_valid)
    np.save("./data/y_test.npy", y_test)
    # integer encode data from tokens perspective
    unknown_token = vocab_size
    tokens_flat = [token for el in X_train for token in el]
    tokens_common = Counter(tokens_flat).most_common()
    tokens_common = dict(tokens_common[:unknown_token-1])
    for i, key in enumerate(tokens_common.keys()):
        tokens_common[key] = i+1
    # encode training set
    X_train = pad_sequences([[tokens_common[el] if el in tokens_common.keys() else unknown_token 
                                      for el in ls] for ls in X_train], maxlen=padding_words)
    # encode validation set
    X_valid = pad_sequences([[tokens_common[el] if el in tokens_common.keys() else unknown_token 
                                      for el in ls] for ls in X_valid],maxlen=padding_words)
    # encode test set
    X_test = pad_sequences([[tokens_common[el] if el in tokens_common.keys() else unknown_token 
                                     for el in ls] for ls in X_test],maxlen=padding_words)
    # save all numpy arrays to file
    np.save("./data/words/X_train.npy", X_train)
    np.save("./data/words/X_valid.npy", X_valid)
    np.save("./data/words/X_test.npy", X_test)
    # save indexing dictionary to file
    with open("./data/words/integer_index_tokens.pickle", "wb") as f:
        pickle.dump(tokens_common, f, protocol=pickle.HIGHEST_PROTOCOL)
    # integer encode data from character perspective
    X_train = ham_train + spam_train
    X_valid = ham_valid + spam_valid
    X_test = ham_test + spam_test
    # clean word index by removing hexadecimal characters
    tokens_flat = [char.lower() for el in X_train for char in el]
    tokens_common = dict(Counter(tokens_flat).most_common())
    for key in list(tokens_common.keys()):
        if tokens_common[key] < 2000:
            tokens_common.pop(key,None)
    for i, key in enumerate(tokens_common):
        tokens_common[key] = i+1
    unknown_token = np.max(list(tokens_common.values()))+1
    # encode training set
    X_train = pad_sequences([[tokens_common[el.lower()] if el.lower() in tokens_common.keys() else unknown_token 
                                      for el in ls] for ls in X_train], maxlen=padding_chars)
    # encode validation set
    X_valid = pad_sequences([[tokens_common[el.lower()] if el.lower() in tokens_common.keys() else unknown_token 
                                      for el in ls] for ls in X_valid],maxlen=padding_chars)
    # encode test set
    X_test = pad_sequences([[tokens_common[el.lower()] if el.lower() in tokens_common.keys() else unknown_token 
                                     for el in ls] for ls in X_test],maxlen=padding_chars)
    # save all numpy arrays to file
    np.save("./data/char/X_train.npy", X_train)
    np.save("./data/char/X_valid.npy", X_valid)
    np.save("./data/char/X_test.npy", X_test)
    # save indexing dictionary to file
    with open("./data/char/integer_index_char.pickle", "wb") as f:
        pickle.dump(tokens_common, f, protocol=pickle.HIGHEST_PROTOCOL)

##############################
# main command call
##############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default = 5000,
                            help="size of vocabulary used in word vector embedding <default:5000>")
    parser.add_argument("--padding", type=int, default = 500,
                        help="maximum length of email padding <default:500>")
    args = parser.parse_args()
    # execute main command
    integerEncode(vocab_size=args.vocab_size, padding=args.padding)