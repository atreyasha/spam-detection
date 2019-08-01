## Notes for model development and comparison

### Architecture
* hypertune runs to find best models, set baselines with individual runs
* bidirectional element to further improve performance

### Word/character/byte embeddings
* extrapolate by using FLAIR pre-trained character/byte embeddings for initialization to overcome unknown words
* initialize with glove embeddings and train from there, for word vectors
* make kernel size and strides larger in character convolutions

### Model comparison
* uniform classifier will already have 50% accuracy, or can be exactly calculated
* compare with bag-of-words on SVM vs. sequential RNN, most likely SVM will exceed
* attain blind-dataset to check robustness of RNNs vs SVMs, use same pre-processing as per enron dataset
* tendency to flag largely misspelled emails, might result in RNN being more robust on blind dataset

### Assumptions
* no name or email address of sender, classify only based on content of email
* assume pre-processed format as provided by 2006 paper authors
* assume language is English and sender address is unknown, as known senders would not be spam-checked
