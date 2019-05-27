"""
For how to use the
collate_fn, see
https://www.codefull.net/2018/11/use-pytorchs-dataloader-with-variable-length-sequences-for-lstm-gru/
"""

import os
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

class Config:
    label_pad_val = -1

def pad_seqs_to_batch(batch):
    """
    Padding sequences to a batch. As a callback function of DataLoader

    Usage:
    >>> loader = DataLoader(..., collate_fn=pad_seqs_to_batch)

    """
    # Let's assume that each element in "batch" is a list of tuple (data, label).
    # Sort the batch in the descending order
    sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    sorted_sequences = []
    sorted_labels = []
    for seq, label in sorted_batch:
        sorted_sequences.append(seq)
        sorted_labels.append(label)

    # Get each sequence and pad it
    sequences_padded = pad_sequence(sorted_sequences)

    # As well as the lables
    labels_padded = pad_sequence(sorted_labels, padding_value=Config.label_pad_val)
    return sequences_padded, labels_padded

class TIMITDataset(Dataset):
    SPLIT_TO_NPZ = {
        'train': "train_set.npz",
        'valid': "validation_set.npz",
        'test': "core_test.npz",
    }

    def __init__(self, root, split):
        """
        Initialize paths, transforms, and so on
        """
        # check input args
        if split not in self.SPLIT_TO_NPZ:
            raise ValueError(
                "Splite must be one of the following, %s" %  ",".join(
                    self.SPLIT_TO_NPZ.keys()))
        # ================================================
        data_npz = os.path.join(root, self.SPLIT_TO_NPZ[split])
        with np.load(data_npz) as f:
            datalist = f['data']
            self.phone_to_idx = f['phone_to_idx']

        # translate the numpy data to tensor
        self.data = list()
        for d in datalist:
            self.data.append({
                'features': torch.Tensor(d['features']),
                'phone_idx': torch.LongTensor(d['phone_idx'])
                })

    def __getitem__(self, index):
        """
        Returns one data pair (source and target).
        """
        d = self.data[index]
        features = d['features']
        label = d['phone_idx']
        return features, label

    def __len__(self):
        """
        Indicate the total size of the dataset
        """
        return len(self.data)

    def get_phone_to_idx(self):
        return self.phone_to_idx

# if __name__ == "__main__":
#     pass
# # create train/val datasets
# trainset = dogDataset(root='./dataset/dogsDataset',
#                       split='train',
#                       transform=train_transform)
# valset = dogDataset(root='./dataset/dogsDataset',
#                     split='test',
#                     transform=val_transform)

# # create train/val loaders
# train_loader = DataLoader(dataset=trainset,
#                           batch_size=16,
#                           shuffle=True,
#                           num_workers=4)
# val_loader = DataLoader(dataset=valset,
#                         batch_size=16,
#                         shuffle=False,
#                         num_workers=4)

# # Get images and labels in a mini-batch of train_loader
# for imgs, lbls in train_loader:
#     print ('Size of image:', imgs.size())  # batch_size*3*224*224
#     print ('Type of image:', imgs.dtype)   # float32
#     print ('Size of label:', lbls.size())  # batch_size
#     print ('Type of label:', lbls.dtype)   # int64(long)
#     break