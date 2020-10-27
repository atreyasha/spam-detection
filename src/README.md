## Comparison of SVM (non-sequential) vs. CNN-LSTM (sequential) models for supervised spam classification

### 1. Overview

In this repository, we will compare a Support Vector Machine (SVM) non-sequential model against a CNN-LSTM sequential model to provide some insight into the effectivitiy and robustness of sequential and non-sequential models in supervised spam classification.

The SVM classifies emails using a bag-of-words representation, while the CNN-LSTM classifies emails using sequential word and/or character vectors; with the possibility of using pre-trained GloVe word and character embeddings. Finally, both sets of models will be compared with relevant evaluation metrics on a test and blind dataset (SMS spam).

### 2. Data acquisition

After cloning this repository, we would need to initialize it with all relevant data; which include the [enron-spam](http://www2.aueb.gr/users/ion/data/enron-spam/) database, [GloVe](https://nlp.stanford.edu/projects/glove/) word embeddings and [SMS-spam](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) dataset. For this, a helper dialogue-based script `data.sh` has been created. To run it, execute the following and follow the corresponding prompts:

```shell
$ ./data.sh
```

### 3. Data pre-processing

Before training and testing our models, we would need to pre-process our data. Please refer to this [readme](/src/docs/pre-processing.md) for more information on data pre-processing.

### 4. Model training

To train our models, we developed scripts that will optimize models and choose the most appropriate ones through a grid-search. Please refer to this [readme](/src/docs/models.md) for more information on our models and their training procedures.

### 5. Model evaluation

To evaluate our models, we developed scripts that output various metrics such as precision, recall, F-1 score and the ROC-AUC. Furthermore, plots of the models and their corresponding evaluation metrics will be made (please install relevant R libraries; see [here](/src/plot_models.R)). Please refer to this [readme](/src/docs/model-evaluation.md) for more information on our evaluation procedures.

### 6. Summary

Finally, for a complete overview of the results/evaluation of this repository, please check our pdf [presentation](/docs/main.pdf) for results and (hopefully nice) visualizations.
