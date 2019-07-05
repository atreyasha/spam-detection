#!/bin/bash
set -e

# move pre-commit hook into local .git folder for activation
cp ./hooks/pre-commit.sample ./.git/hooks/pre-commit

# download data
cd ./src/data/enron && ./enron_spam.sh && cd ../../..
