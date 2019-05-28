#!/bin/bash
#
#  Script to pre-processing TIMIT dataset
#
set -e

export PYTHONPATH=`pwd`
mkdir -p ./data/raw

# 1. Calcuate mfcc featrues from the whole training dataset (3696  sentances)
python scripts/get_train_dataset.py
# 2. Select 184 sentences as validation set
python scripts/get_val_sentences.py
# 3. Split the sentences according to the list of sentences
#    Normalize the training and validation set.
#    Normalizer is calculated from splitted training set (3152 sentances)
python scripts/split_val_train.py
# 4. Calcuate mfcc featrues from core test and normalize it with the
#    normalizer in stage 3.
python scripts/get_core_test.py
