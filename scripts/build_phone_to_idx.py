import pandas as pd
import json

if __name__ == "__main__":
    df = pd.read_csv("data/map/phones.60-48-39.map", sep='\t', header=None)
    idx_to_phone = df[0].to_dict()

    phone_to_idx = dict()
    for k, v in idx_to_phone.items():
        phone_to_idx[v] = k

    with open('data/raw/phone_to_idx.json', 'w') as f:
        json.dump(phone_to_idx, f, indent=2)