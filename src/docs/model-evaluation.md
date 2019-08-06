### Model evaluation

After performing grid-searches on the models, we would need to evaluate our best models with various metrics such as precision, recall, F1 and ROC-AUC. This is particularly important for the task of spam classification, where the costs of misclassifying ham emails are generally much higher than those of misclassifying spam emails. For this, we provide two scripts; namely `model_evaluation.py` and `model_visualization.py`.

#### 1. Model probabilities and decision functions

`model_raw_values.py` produces model probabilities and decision functions on the test and blind dataset.

```
usage: model_raw_values.py [-h] [--padding-tokens PADDING_TOKENS]
                           [--padding-char PADDING_CHAR] -p PICKLE

optional arguments:
  -h, --help            show this help message and exit
  --padding-tokens PADDING_TOKENS
                        maximum length of email padding for tokens
                        <default:500>
  --padding-char PADDING_CHAR
                        maximum length of email padding for characters
                        <default:1000>

required named arguments:
  -p PICKLE, --pickle PICKLE
                        pickle directory name for stored model, or input 'all'
                        to run on all models
```

An example of running this would be:

```shell
$ python3 model_raw_values.py -p all
```

#### 2. Model evaluation and visualization

After running the script above, we can then visualize some of our results. `model_eval_vis.py` performs optimal threshold analysis and provides the best threshold to maximize recall on ham emails while still preserving precision. If this optimization cannot be met, then it attempts to find the next best threshold.

The script also produces classification reports and ROC-AUC scores on the test dataset based on various thresholds. In terms of visualization, this script produces charts for words-based relative importance analysis from the SVM, precision-recall curves for all models based on varying thresholds, and ROC curves for the test and blind dataset. Note that this script calls `R` to plot some of the charts.

```
usage: model_eval_vis.py [-h] [--padding-tokens PADDING_TOKENS]
                         [--padding-char PADDING_CHAR] -p PICKLE

optional arguments:
  -h, --help            show this help message and exit
  --padding-tokens PADDING_TOKENS
                        maximum length of email padding for tokens
                        <default:500>
  --padding-char PADDING_CHAR
                        maximum length of email padding for characters
                        <default:1000>

required named arguments:
  -p PICKLE, --pickle PICKLE
                        pickle directory name for stored model, or input 'all'
                        to run on all models
```

An example of running this would be:

```shell
$ python3 model_eval_vis.py -p all
```

A sample output of the combined chart for all models can be seen below:

<p align="center">
<img src="/img/combined.png" width="800">
</p>
