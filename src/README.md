## Spam detection using LSTM-CNN architecture (Enron-spam dataset)

### Sequence Encoding

Before running the models, we would need to encode the enron-spam preprocessed data into integers. For this, we define a helper function in `data_encode.py`:

```
usage: data_encode.py [-h] [--vocab-size VOCAB_SIZE] [--padding PADDING]

optional arguments:
  -h, --help            show this help message and exit
  --vocab-size VOCAB_SIZE
                        size of vocabulary used in word vector embedding
                        <default:5000>
  --padding PADDING     maximum length of email padding <default:500>
```

Currently, we only support word integer encoding, but the development of character-based encodings is underway.

In order to encode the enron-spam emails, execute the following command:

```shell
$ python3 data_encode.py
```
