## Comparison of LSTM-CNN (sequential) vs. SVM (non-sequential) models for supervised spam classification

### Table of Contents

1. [Data preprocessing](#Data-preprocessing)
2. Model training
3. Model comparisons

### 1. Data preprocessing

Before running the models, we would need to preprocess our text based data for model training. 

#### 1.1. Integer encoding for CNN-LSTM

To preprocess data for our sequential model, we define a helper function in `integer_encode.py`:

```
usage: integer_encode.py [-h] [--vocab-size VOCAB_SIZE]
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

Running this function will encode the enron-spam dataset as integer-based tokens and characters. The two sets of encodings will be saved in the `./data/rnn` directory. An example of running this function is show below:

```shell
$ python3 integer_encode.py
```
#### 1.2. Bag-of-words encoding for SVM

To preprocess data for our non-sequential model, we define a helper function in `bag_words.py`:

```
usage: bag_words.py [-h] [--vocab-size VOCAB_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --vocab-size VOCAB_SIZE
                        size of vocabulary used in word vector embedding
                        <default:5000>
```

Running this function will encode the enron-spam dataset in a bag-of-words format. This representation of encodings will be saved in the `./data/svm` directory. An example of running this function is show below:

```shell
$ python3 bag_words.py
```

Note: Further development underway :snail:
