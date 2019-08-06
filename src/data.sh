#!/bin/bash
set -e

# download enron spam data and deploy for usage
read -rep "download and deploy enron-spam data? (y/n): " ans
if [ $ans == "y" ]; then
	cd ./data/enron && ./enron_spam.sh && cd ../..
fi

# download and deploy glove word embeddings
read -rep "download and deploy glove(6B) data? (y/n): " ans
if [ $ans == "y" ]; then
	cd ./data/glove && wget http://nlp.stanford.edu/data/glove.6B.zip && unzip glove.6B.zip && cd ../..
fi

# download blind sms spam dataset
read -rep "download and deploy SMS spam blind data? (y/n): " ans
if [ $ans == "y" ]; then
	cd ./data/blind && wget https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip && unzip smsspamcollection.zip && cd ../..
fi
