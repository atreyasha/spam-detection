#!/bin/bash

# download files to directory
for i in {1..6}
	do wget -N "http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/enron${i}.tar.gz"
done

# extract all text files
files=$(ls|grep tar)
for f in $files
	do tar -zxvf $f
done
