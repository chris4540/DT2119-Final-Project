"""
1. Split data
2. normalization
3. save out related models and data
"""
import numpy as np
import json
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

class Config:
    full_train_data = "data/raw/full_traindata.npz"
    val_sentence_json = "data/val_sent.json"
    scaler_pkl = "data/feature_scaler.pkl"
    val_dump = "data/validation_set.npz"
    train_dump = "data/train_set.npz"

if __name__ == "__main__":
    with open(Config.val_sentence_json, 'r') as f:
        val_sents = json.load(f)

    # load full set of training data
    data = np.load(Config.full_train_data)
    phone_to_idx = data['phone_to_idx'].item()
    full_traindata = data['data']
    # ======================================================
    train_data = []
    val_data = []
    for d in full_traindata:
        wavfile = d['file']
        if wavfile in val_sents:
            val_data.append(d)
        else:
            train_data.append(d)
    # ========================================================
    # train the StandardScaler with only training data
    scaler = StandardScaler()
    print("Training StandardScaler...")
    for d in tqdm(train_data):
        scaler.partial_fit(d['features'])

    for d in train_data:
        d['features'] = scaler.transform(d['features'])

    for d in val_data:
        d['features'] = scaler.transform(d['features'])

    # =======================================================
    # save the scalar for transforing test data
    print("Saving scaler model to %s ..." % Config.scaler_pkl)
    joblib.dump(scaler, Config.scaler_pkl)

    # save the validation data
    kwargs = {
        'data': val_data,
        'phone_to_idx': phone_to_idx,
    }
    print("Saving validation data: %s ..." % Config.val_dump)
    np.savez(Config.val_dump, **kwargs)

    # save the training data
    kwargs = {
        'data': train_data,
        'phone_to_idx': phone_to_idx,
    }
    print("Saving training data: %s ..." % Config.train_dump)
    np.savez(Config.train_dump, **kwargs)

