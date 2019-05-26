import os
from preprocess import TIMITFeatureExtractor
from tqdm import tqdm
import numpy as np

def prase_phone_to_idx(phone, phone_to_idx_dict):
    unique_phones = np.unique(phone)
    for ph in unique_phones:
        if ph not in phone_to_idx_dict:
            phone_to_idx_dict[ph] = len(phone_to_idx_dict)

    phone_idx_arr = np.vectorize(phone_to_idx_dict.get)(phone)
    return phone_idx_arr, phone_to_idx_dict

if __name__ == "__main__":
    # config
    folder = os.path.join("TIMIT", "TRAIN")
    dump_file_name = "traindata.npz"
    # ===================================================
    phone_to_idx = dict()
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
            phone_idxs, phone_to_idx = prase_phone_to_idx(phone, phone_to_idx)
            data.append({
                'features': extracted['features'],
                'phone_idx': phone_idxs,
                'file': extracted['file']
            })
            cnt += 1

            if cnt % 500 == 0:
                print("Processed %d data...." % cnt)

    # saving
    kwargs = {
        'data': data,
        'phone_to_idx': phone_to_idx
    }
    np.savez(dump_file_name, **kwargs)