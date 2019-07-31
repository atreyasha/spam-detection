## Notes for model development and comparison

### Architecture
* hypertune runs to find best models, set baselines with individual runs
* bidirectional and attention elements can further improve performance
* attention can be dropped and character/word embeddings to keep

### Word/character/byte embeddings
* utilize character/byte embeddings and check potential/generality, avoids spelling mistakes
* extrapolate by using FLAIR pre-trained character embeddings for initialization
* initialize with glove embeddings and train from there, for word vectors
* possible to also use stacked character and word embeddings for robustness and semantic scope
* make kernel size and strides larger in character convolutions

### Model comparison
* uniform classifier will already have 50% accuracy, or can be exactly calculated
* compare with bag-of-words on SVM vs. sequential RNN, most likely SVM will exceed
* attain blind-dataset to check robustness of RNNs vs SVMs, use same pre-processing as per enron dataset
* tendency to flag largely misspelled emails, might result in RNN being more robust on blind dataset

### Extras
* put error message if model cannot be compiled due to conflicting vocab size

### Assumptions
* no name or email address of sender, classify only based on content of email
* assume pre-processed format as provided by 2006 paper authors
* assume language is English and sender address is unknown, as known senders would not be spam-checked
