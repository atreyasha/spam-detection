## Comparison of sequential and non-sequential models for supervised spam classification (Enron-spam dataset)

The central objective of this repository is to shed light and provide evidence on whether sequential learning is inherently important for the task of supervised spam classification. This is done by implementing a sequential CNN-LSTM model and comparing it with a non-sequential SVM classifier. Aspects such as generalization accuracy and robustness of these models will be explored in this repository.

The models will be trained on the pre-processed `enron-spam` dataset (http://www2.aueb.gr/users/ion/data/enron-spam/), consisting of ~34k instances of "ham" and spam emails.

To initialize this repository, it is recommended to enable a pre-commit hook which updates python dependencies in `requirements.txt`. It is also recommended to automatically download the enron-spam dataset; both of which can be done by executing the following:

```shell
$ ./init.sh
```

Further information on the model and results can be found in the `/src` directory and the corresponding [README.md](/src/README.md)

## Citation

Below is the BibTeX citation for Metsis et al. 2006, which is the paper published alongside the pre-processed enron dataset in this repository.

```
@inproceedings{metsis2006spam,
  title={Spam filtering with naive bayes-which naive bayes?},
  author={Metsis, Vangelis and Androutsopoulos, Ion and Paliouras, Georgios},
  booktitle={CEAS},
  volume={17},
  pages={28--69},
  year={2006},
  organization={Mountain View, CA}
}
```

## Author

Atreya Shankar, Cognitive Systems 2018
