## Spam detection using LSTM-CNN architecture (Enron-spam dataset)

### Sequence Encoding

Before running the models, we would need to encode the enron-spam preprocessed data into integers. For this, we define a helper function in `data_encode.py`:

```
usage: data_encode.py [-h] [--vocab-size VOCAB_SIZE]
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

Running this function will encode the enron-spam dataset as integer-based tokens and characters. The two sets of encodings will be saved in the `./data` directory. An example of running this function is show below:

```shell
$ python3 data_encode.py
```
