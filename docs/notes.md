## Notes for model development

### Architecture
* hypertune runs to find best models for subject and body separately
* then combine to see if subject and body together can outperform
* set baselines with individual runs
* can compare with model based on character embeddings
* smaller vocabulary of just pure possible elements with spaces included
* but very large maximum sequence length

### Word embeddings
* utilize character embeddings and check potential/generality
* if good, can extrapolate by using FLAIR pre-trained character embeddings for initialization
* initialize with glove embeddings and train from there, for word vectors
* make two sets of word vectors, for body and subject
* use NER tagger to remove named entities for more generality
* think of unknown word handling, maybe skip or add unknown vector
* possible to use half-known words and half-unknown words both initialized with pre-trained embeddings (only known will be trained)
* possible to also use stacked character and word embeddings for robustness and semantic scope

### Model comparison
* uniform classifier will already have 50% accuracy, or can be exactly calculated
* compare with bag-of-words with SVM
* what can be done to make the model more general

### Additional
* try attention-based mechanism
* maybe add language as additional input
* only use spam filter if unknown address or not yourself
* put error message if model cannot be compiled due to conflicting vocab size

### Interesting and challenging
* check if spam model can actually re-generate spam text, which would be interesting
* would need a better technique of unknown word handling due to them turning up in generated text
* maybe can add multiple unknown tokens to spice things up
