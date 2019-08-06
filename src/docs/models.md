### Model training

After pre-processing our data, we can then train/optimize our models with grid-searches over various hyperparameters.

#### 1. Support Vector Machine (SVM)

Executing `train_svm.py` will conduct a grid-search to train a SVM using scikit-learn's SGDClassifier using mini-batch Stochastic Gradient Descent (MB-SGD). Early-stopping based on validation dataset performance is included. The SGDClassifier's default kernel is the linear kernel, however an option can be passed to use the RBF kernel with a kernel approximation (RBFSampler). Further details can be found below:

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

#### 2. CNN-LSTM

In the CNN-LSTM set of models, we provide three different approaches. The first two approaches work by using either purely word or character encodings and embeddings. These can be visualized through the keras model below:

<img src="/img/model.png" width="500">

An advantage of a character-based approach is that it mitigates the unknown token issue; which could add to model robustness. However, character sequences lose semantic meaning of written text, so it might be viable to combine both versions. For this, we propose another model where character and word sequences are both used in classification. This is visualized below:

<img src="/img/model_combined.png" width="500">

Finally, all three approaches can have the word/character embeddings randomly initialized or setup with the GloVe embeddings.

Executing `train_rnn.py` will conduct a grid-search to train and test a CNN-LSTM over a validation/test set. Further details can be found below:

```
usage: train_rnn.py [-h] [--subtype SUBTYPE] [--pre-trained-embeddings]
                    [--grid-search] [--single-run] [--plot] [--name NAME]

optional arguments:
  -h, --help            show this help message and exit
  --subtype SUBTYPE     which model subtype to use; either 'words', 'char' or
                        'all' <default:'words'>
  --pre-trained-embeddings
                        option to use pre-trained word/character embeddings,
                        disabled by default
  --grid-search         option to conduct grid-search, enabled by default
  --single-run          option to conduct single run based on default
                        hyperparameters, disabled by default
  --plot                option for plotting keras model, disabled by default
  --name NAME           if --plot option is chosen, this provides name of the
                        model image <default:'model'>
```

Best models and log files for grid searches will be saved under the `./pickles` directory. An example of running the file is shown below:

```
$ python3 train_rnn.py
```
