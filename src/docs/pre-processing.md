### Data preprocessing

Before running the models, we would need to preprocess our text based data for model training.

#### 1. Bag-of-words encoding for SVM

To preprocess data for our non-sequential model, we define a helper function in `bag_words.py`. This function reads in emails data and conducts the following pre-processing procedures:

```
tokenizing -> POS-tagging -> removing stop words -> lemmatizing with pos-tags
```

This creates a clean set of words for our bag-of-words approach. We fill the bag-of-words matrix with word frequencies and later normalize the rows to get the bag-of-words in a probability representation.

```
usage: bag_words.py [-h] [--vocab-size VOCAB_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --vocab-size VOCAB_SIZE
                        size of vocabulary used in bag-of-words encoding
                        <default:5000>
```

Running this function will encode the enron-spam dataset in a bag-of-words format. This representation of encodings will be saved in the `./data/svm` directory. An example of running this function is shown below:

```shell
$ python3 bag_words.py
```

#### 2. Sequence encoding for CNN-LSTM

To preprocess data for our sequential model, we define a helper function in `sequence_encode.py`. This process tokenizes the words and, when desired, also maps them to sequential characters. Next, a word/character to integer mapping is created and the sequences are integer-encoded as per the mapping. Lastly, the sequences are padded to the provided maximum padding length, which can then be fed in to embedding layers in Keras.

```
usage: sequence_encode.py [-h] [--vocab-size VOCAB_SIZE]
                         [--padding-tokens PADDING_TOKENS]
                         [--padding-char PADDING_CHAR]

optional arguments:
  -h, --help            show this help message and exit
  --vocab-size VOCAB_SIZE
                        size of vocabulary used in word vector embedding
                        <default:5000>
  --padding-tokens PADDING_TOKENS
                        maximum length of email padding for tokens
                        <default:500>
  --padding-char PADDING_CHAR
                        maximum length of email padding for characters
                        <default:1000>
```

Running this function will encode the enron-spam dataset as integer-based tokens and characters. The two sets of encodings will be saved in the `./data/rnn` directory. An example of running this function is shown below:

```shell
$ python3 sequence_encode.py
```

#### 3. GloVe embedded matrix generation

In order to test/use the GloVe word and character embeddings, we would need to match the GloVe words to our own vocabulary. `embedding_matrix_gen.py` automatically does this for us and also approximates GloVe character embeddings using a methodology from [`char-embeddings`](https://github.com/minimaxir/char-embeddings), which is distributed under the MIT [License](../../THIRD_PARTY_NOTICES.txt). To create the embedding matrices, run the following:

```shell
$ python3 embedding_matrix_gen.py
```

#### 4. Note on data consistency

Due to the large temporal resources required for grid-searches, we omit K-fold cross-validation. Instead, we select balanced train/valid/test datasets and utilize them consistently for training and evaluating both sets of models. Although this would not mitigate data bias due to the chosen splits, this would still ensure that performances on these split datasets are at least comparable.
