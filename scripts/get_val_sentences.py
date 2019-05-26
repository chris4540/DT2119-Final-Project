"""
Get the validation set of sentences

1. calculate the number of sentences read by males and females.
2. distribute them to each dialect
3. draw # of sentences // 8 in each dialect
4. save the list of sentences to a json file for review
"""
import numpy as np
import os
import json

# don't move this number
np.random.seed(40)

class Config:
    full_train_data = "data/raw/full_traindata.npz"
    val_sentence_json = "data/val_sent.json"
    num_val_sent = 184
    num_male_val_sent = int(0.7 * num_val_sent)
    num_female_val_sent =  num_val_sent - num_male_val_sent
    p_train = .8
    p_val = 1 - p_train

def get_speaker_info(wavefile_path):
    path = os.path.normpath(wavefile_path)
    info = path.split(os.sep)
    speaker_id = info[-2]
    gender = speaker_id[0]
    dialect = info[2]
    return {
        'gender': gender,
        'dialect': dialect
    }

if __name__ == "__main__":
    data = np.load(Config.full_train_data)['data']
    val_sentence = list()
    # ===================================================================
    dialect = 'DR1'
    cur_dialect = 'DR1'
    n_male_sent_to_draw = Config.num_male_val_sent // 8
    n_fmale_sent_to_draw = Config.num_female_val_sent // 8
    for d in data:

        if cur_dialect != dialect:
            dialect = cur_dialect
            n_male_sent_to_draw = Config.num_male_val_sent // 8
            n_fmale_sent_to_draw = Config.num_female_val_sent // 8

        if len(val_sentence) < Config.num_val_sent:
            c = np.random.choice(['train', 'val'], p=[Config.p_train, Config.p_val])
            if c == 'train' :
                continue

            info = get_speaker_info(d['file'])
            gender = info['gender']
            cur_dialect = info['dialect']
            if gender == "M" and n_male_sent_to_draw > 0:
                n_male_sent_to_draw -= 1
                val_sentence.append(d['file'])
            elif gender == "F" and n_fmale_sent_to_draw > 0:
                n_fmale_sent_to_draw -= 1
                val_sentence.append(d['file'])
        else:
            # enough validation set, break it down
            break

    assert len(val_sentence) == Config.num_val_sent

    # save the list as a json file
    print("Saving the validation sentence list to %s" % Config.val_sentence_json)
    with open(Config.val_sentence_json, 'w') as f:
        json.dump(val_sentence, f, indent=2)