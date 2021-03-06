"""
Generate the training dataset.

The number of phone should be 48 after using this script.

Read data:
>>> data = np.load('data/raw/full_traindata.npz')
>>> data['phone_to_idx'].item()  # the mapping
>>> traindata = data['data'] # training features


Prerequisite:
    data/raw/phone_to_idx.json
Generate by:
    ipython scripts/build_phone_to_idx.json can help to build the json
"""

import os
from preprocess import TIMITFeatureExtractor
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
from utils import map_phone_to_idx

class Config:
    dump_file_name = "data/raw/full_traindata.npz"
    phone_map_tsv = "data/map/phones.60-48-39.map"
    folder = os.path.join("TIMIT", "TRAIN")

def extract_featurs_from(folder, phone_to_idx):
    # ===================================================
    data = list()
    cnt = 0
    for root, dirs, files in os.walk(folder):
        for f in files:
            fname = os.path.join(root, f)
            # skip if not a sound file
            if not f.endswith(".WAV"):
                continue
            # skip if SA sentances
            if f.startswith("SA"):
                continue

            ext = TIMITFeatureExtractor(fname)
            extracted = ext.extract()

            phone = extracted['phone']

            # drop q phone
            idxs = np.argwhere(phone == 'q')
            phone = np.delete(phone, idxs)
            features = np.delete(extracted['features'], idxs, axis=0)
            assert len(phone) == features.shape[0]

            if '0' in phone:
                print(fname)
                raise IOError("Encounter 0 phone")

            phone_idxs = map_phone_to_idx(phone, phone_to_idx)
            data.append({
                'features': features,
                'phone_idx': phone_idxs,
                'file': extracted['file']
            })
            cnt += 1

            if cnt % 500 == 0:
                print("Processed %d data...." % cnt)

    return data

if __name__ == "__main__":
    # load the mapping
    df = pd.read_csv(Config.phone_map_tsv, sep="\t", index_col=0)
    df = df.dropna()
    df = df.drop('eval', axis=1)
    train_phn_idx = {k: i for i, k in enumerate(df['train'].unique())}
    df['train_idx'] = df['train'].map(train_phn_idx)
    phone_to_idx = df['train_idx'].to_dict()


    data = extract_featurs_from(Config.folder, phone_to_idx)
    print("Writing training data to %s ...." % Config.dump_file_name)
    # saving
    kwargs = {
        'data': data,
        'phone_to_idx': train_phn_idx
    }
    np.savez(Config.dump_file_name, **kwargs)
