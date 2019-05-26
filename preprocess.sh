#!/bin/bash
#
#  Script to pre-processing TIMIT dataset
#
#
set -e

export PYTHONPATH=`pwd`
mkdir -p ./data/raw

# Obtain whole training dataset
python scripts/get_train_dataset.py
# Selecting from it
python scripts/get_val_sentences.py
# Try to split them
python scripts/split_val_train.py
# Prepare the core test
python scripts/get_core_test.py
