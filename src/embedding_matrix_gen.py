#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import pickle
import numpy as np
from tqdm import tqdm
from glob import glob

def charEmbeddingGen():
    # adapted from https://github.com/minimaxir/char-embeddings
    file_path = glob("./data/glove/*d.txt")
    for file in tqdm(file_path):
        vectors = {}
        with open(file, 'r') as f:
            for line in f:
                line_split = line.split()
                vec = np.array(line_split[1:], dtype=float)
                word = line_split[0]
                for char in word:
                    if ord(char) < 128:
                        if char in vectors:
                            vectors[char] = (vectors[char][0] + vec,
                                             vectors[char][1] + 1)
                        else:
                            vectors[char] = (vec, 1)
        # write to character embeddings file
        base_name = re.sub(".txt","",os.path.dirname(file) +"/"+ os.path.basename(file)) + '_char.txt'
        with open(base_name, 'w') as f:
            for word in vectors:
                avg_vector = np.round(
                    (vectors[word][0] / vectors[word][1]), 6).tolist()
                f.write(word + " " + " ".join(str(x) for x in avg_vector) + "\n")

def embeddingMatrix():
    # adapted from https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
    with open("./data/rnn/words/integer_index_tokens.pickle","rb") as f:
        index = pickle.load(f)
    # write embedding matrices for words
    file_path = glob("./data/glove/*d.txt")
    for file in file_path:
        embedding_vector_size = int(re.sub("d","",re.findall(r"\d+d",file)[0]))
        embeddings_index = dict()
        with open(file, "r") as f:
            for line in tqdm(f):
                values = line.split()
                word = values[0]
                if word in index:
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
        # create a weight matrix for words in training docs
        embedding_matrix = np.zeros((len(index.keys())+1, embedding_vector_size))
        # initialize null token
        embedding_matrix[0] = np.random.normal(size=(embedding_vector_size,))
        for word, i in index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                embedding_matrix[i] = np.random.normal(size=(embedding_vector_size,))
        np.save(re.sub(".txt","_word_emb.npy",file),embedding_matrix)
    # write embedding matrices for characters
    with open("./data/rnn/char/integer_index_char.pickle","rb") as f:
        index = pickle.load(f)
    # write embedding matrices for words
    file_path = glob("./data/glove/*char.txt")
    for file in file_path:
        embedding_vector_size = int(re.sub("d","",re.findall(r"\d+d",file)[0]))
        embeddings_index = dict()
        with open(file, "r") as f:
            for line in tqdm(f):
                values = line.split()
                word = values[0]
                if word in index:
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
        # create a weight matrix for words in training docs
        embedding_matrix = np.zeros((len(index.keys())+1, embedding_vector_size))
        # initialize null token
        embedding_matrix[0] = np.random.normal(size=(embedding_vector_size,))
        for word, i in index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                embedding_matrix[i] = np.random.normal(size=(embedding_vector_size,))
        np.save(re.sub(".txt","_emb.npy",file),embedding_matrix)

if __name__ == "__main__":
    charEmbeddingGen()
    embeddingMatrix()