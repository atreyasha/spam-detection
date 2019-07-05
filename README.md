## Spam detection using LSTM-CNN architecture (Enron-spam dataset)

This repository contains source code for a supervised LSTM-CNN deep model which can classify emails into "ham" or spam. The deep learning model is trained on the pre-processed `enron-spam` dataset (http://www2.aueb.gr/users/ion/data/enron-spam/), consisting of ~34k instances of "ham" and spam emails.

To initialize this repository, it is recommended to enable a pre-commit hook which updates python dependencies in `requirements.txt`. It is also recommended to automatically download the enron-spam dataset; both of which can be done by executing the following:

```shell
$ ./init.sh
```

Further information on the model and results can be found in the `/src` directory and the corresponding [README.md](/src/README.md)
