#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################
# import dependencies
##############################

import nltk
import pickle
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from keras.preprocessing.text import text_to_word_sequence
from sklearn.model_selection import train_test_split

##############################
# define key functions
##############################

# modified, source https://stackoverflow.com/a/15590384
def _get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return "a"
    elif treebank_tag.startswith('V'):
        return "v"
    elif treebank_tag.startswith('N'):
        return "n"
    elif treebank_tag.startswith('R'):
        return "r"
    else:
        return "n"
    
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
    store = []
    try:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
    except LookupError:
        nltk.download("wordnet")
        lem = nltk.stem.wordnet.WordNetLemmatizer()
    try:
        nltk.corpus.stopwords.words('english')
    except LookupError:
        nltk.download("stopwords")
    for el in tqdm(obj):
        int_store = []
        res = nltk.pos_tag(text_to_word_sequence(el))
        for tup in res:
            if tup[0].isalpha() and tup[0] not in nltk.corpus.stopwords.words('english'):
                int_store.append(lem.lemmatize(tup[0], _get_wordnet_pos(tup[1])))
        store.append(int_store)
    return store

def bagging(dataset,word_dict,unknown_token=None):
    if unknown_token == None:
        unknown_token = len(word_dict.keys())
    hold = np.zeros(shape=(len(dataset),len(word_dict.keys())+1),dtype=int)
    for i, ls in enumerate(dataset):
        local_dict = dict(Counter(ls))
        for key in local_dict.keys():
            if key in word_dict:
                hold[i,word_dict[key]] = local_dict[key]
            else:
                hold[i,unknown_token] = local_dict[key]
    return hold

def bagWords(vocab_size=5000):
    ham = [filename for filename in Path('./data/enron').glob('**/*ham.txt')]
    spam = [filename for filename in Path('./data/enron').glob('**/*spam.txt')]
    hamCorpus = read_order(ham)
    spamCorpus = read_order(spam)
    label_ham = [-1 for _ in range(len(hamCorpus))]
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
    np.save("./data/svm/y_train.npy", y_train)
    np.save("./data/svm/y_valid.npy", y_valid)
    np.save("./data/svm/y_test.npy", y_test)
    # integer encode data from tokens perspective
    unknown_token = vocab_size-1
    tokens_flat = [token for el in X_train for token in el]
    tokens_common = Counter(tokens_flat).most_common()
    tokens_common = dict(tokens_common[:unknown_token])
    for i, key in enumerate(tokens_common.keys()):
        tokens_common[key] = i
    # encode training set
    X_train = bagging(X_train,tokens_common)
    X_valid = bagging(X_valid,tokens_common)
    X_test = bagging(X_test,tokens_common)
    # save all numpy arrays to file
    np.save("./data/svm/words/X_train.npy", X_train)
    np.save("./data/svm/words/X_valid.npy", X_valid)
    np.save("./data/svm/words/X_test.npy", X_test)
    # save indexing dictionary to file
    with open("./data/svm/words/integer_index_tokens.pickle", "wb") as f:
        pickle.dump(tokens_common, f, protocol=pickle.HIGHEST_PROTOCOL)
    
##############################
# main command call
##############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default = 5000,
                            help="size of vocabulary used in bag-of-words encoding <default:5000>")
    args = parser.parse_args()
    # execute main command
    bagWords(vocab_size=args.vocab_size)