## Notes for model development and comparison

### Evaluations/Model comparison
* uniform classifier will already have 50% accuracy, or can be exactly calculated
* compare with bag-of-words on SVM vs. sequential RNN, most likely SVM will exceed
* attain blind-dataset to check robustness of RNNs vs SVMs, use same pre-processing as per enron dataset
* tendency to flag largely misspelled emails, might result in RNN being more robust on blind dataset
* add option to print classification report for train and valid datasets during evaluation period

### Assumptions
* no name or email address of sender, classify only based on content of email
* assume pre-processed format as provided by 2006 paper authors
* assume language is English and sender address is unknown, as known senders would not be spam-checked

### Preliminary Conclusions:
* overfitting for RNN based on character and word sequences
* it can handle spelling mistakes but fits on types of text very well
* SVM's token capabilities can generalize on texts as long as some known words are present
* sequences are important, but word content is more important for spam classification
* effect of spelling mistakes is less important compared to presence of words
