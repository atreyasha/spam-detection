## Comparison of SVM (non-sequential) vs. CNN-LSTM (sequential) models for supervised spam classification

### 1. Data preprocessing

Before running the models, we would need to preprocess our text based data for model training.

#### 1.1. Bag-of-words encoding for SVM

To preprocess data for our non-sequential model, we define a helper function in `bag_words.py`:

```
usage: bag_words.py [-h] [--vocab-size VOCAB_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --vocab-size VOCAB_SIZE
                        size of vocabulary used in bag-of-words encoding
                        <default:5000>
```

Running this function will encode the enron-spam dataset in a bag-of-words format. This representation of encodings will be saved in the `./data/svm` directory. An example of running this function is show below:

```shell
$ python3 bag_words.py
```

#### 1.2. Sequence encoding for CNN-LSTM

To preprocess data for our sequential model, we define a helper function in `sequence_encode.py`:

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

Running this function will encode the enron-spam dataset as integer-based tokens and characters. The two sets of encodings will be saved in the `./data/rnn` directory. An example of running this function is show below:

```shell
$ python3 sequence_encode.py
```

### 2. Model training

The following scripts/functions conduct grid-searches over various hyperparameters to train a SVM and CNN-LSTM.

#### 2.1. Support Vector Machine (SVM)

Executing `train_svm.py` will conduct a grid-search to train a SVM using scikit-learn's SGDClassififer using mini-batch Stochastic Gradient Descent (SGD). Early-stopping based on validation dataset performance is included. The SGDClassifier's default kernel is the linear kernel, however an option can be passed to use the RBF kernel with a kernel approximation. Further details can be found below:

```
usage: train_svm.py [-h] [--epochs EPOCHS] [--patience PATIENCE]
                    [--kernels KERNELS]

optional arguments:
  -h, --help           show this help message and exit
  --epochs EPOCHS      maximum number of epochs for training <default:50>
  --patience PATIENCE  patience for early stopping <default:5>
  --kernels KERNELS    which kernels to search over, either 'linear', 'rbf' or
                       'all' <default:'linear'>
```

Best models and log files for grid searches will be saved under the `./pickles` directory. An example of running the file is shown below:

```
$ python3 train_svm.py
```

#### 2.2. CNN-LSTM

Executing `train_rnn.py` will conduct a grid-search to train and test a CNN-LSTM over a validation set. Further details can be found below:

```
usage: train_rnn.py [-h] [--subtype SUBTYPE] [--grid-search] [--single-run]
                    [--plot] [--name NAME]

optional arguments:
  -h, --help         show this help message and exit
  --subtype SUBTYPE  which model subtype to use; either 'words', 'char' or
                     'all' <default:'words'>
  --grid-search      option to conduct grid-search, enabled by default
  --single-run       option to conduct single run based on default
                     hyperparameters, disabled by default
  --plot             option for plotting keras model, disabled by default
  --name NAME        if --plot option is chosen, this provides name of the
                     model image <default:'model'>
```

Best models and log files for grid searches will be saved under the `./pickles` directory. An example of running the file is shown below:

```
$ python3 train_rnn.py
```

Note: Further development underway :snail:
