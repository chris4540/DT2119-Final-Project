"""
Script to generate core test
"""
import os
import json
import numpy as np
from utils import map_phone_to_idx
from preprocess import TIMITFeatureExtractor

class Config:
    dump_file_name = "data/core_test.npz"
    phone_to_idx_json = "data/raw/phone_to_idx.json"
    timit_test_folder = "./TIMIT/TEST"

# ==============================================================================
# This list is edited from TIMIT/README.DOC
core_test_spkr = [
    #       Male         Male       Female
        "DR1/MDAB0", "DR1/MWBT0", "DR1/FELC0",
        "DR2/MTAS1", "DR2/MWEW0", "DR2/FPAS0",
        "DR3/MJMP0", "DR3/MLNT0", "DR3/FPKT0",
        "DR4/MLLL0", "DR4/MTLS0", "DR4/FJLM0",
        "DR5/MBPM0", "DR5/MKLT0", "DR5/FNLP0",
        "DR6/MCMJ0", "DR6/MJDH0", "DR6/FMGD0",
        "DR7/MGRT0", "DR7/MNJM0", "DR7/FDHC0",
        "DR8/MJLN0", "DR8/MPAM0", "DR8/FMLD0",
]

if __name__ == "__main__":
    # read the phone_to_idx json file
    with open(Config.phone_to_idx_json, 'r') as f:
        phone_to_idx = json.load(f)

    # ==========================================================
    cnt = 0
    core_test_data = list()
    # ==========================================================
    for folder in core_test_spkr:
        spkr_folder = os.path.join(Config.timit_test_folder, folder)
        for root, dirs, files in os.walk(spkr_folder):
            for f in files:
                if not f.endswith(".WAV"):
                    continue
                # skip if SA sentances
                if f.startswith("SA"):
                    continue

                fname = os.path.join(root, f)

                # Extract data
                ext = TIMITFeatureExtractor(fname)
                extracted = ext.extract()

                phone = extracted['phone']
                if '0' in phone:
                    raise IOError("Encounter 0 phone for the file: " + fname)

                phone_idxs = map_phone_to_idx(phone, phone_to_idx)
                core_test_data.append({
                    'features': extracted['features'],
                    'phone_idx': phone_idxs,
                    'file': extracted['file']
                })
                cnt += 1
    print("Extracted %d data" % cnt)
    # ============================================================
    print("Writing core test data to %s ...."% Config.dump_file_name)
    # saving
    kwargs = {
        'data': core_test_data
    }
    np.savez(Config.dump_file_name, **kwargs)
