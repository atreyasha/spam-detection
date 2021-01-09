#!/bin/bash
set -e
# move pre-commit hook into local .git folder for activation
cp ./hooks/pre-commit ./.git/hooks/pre-commit
