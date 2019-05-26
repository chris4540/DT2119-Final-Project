"""
https://github.com/yunjey/seq2seq-dataloader/blob/master/data_loader.py
https://github.com/keishinkickback/Pytorch-RNN-text-classification/blob/master/dataloader.py
https://www.codefull.net/2018/11/use-pytorchs-dataloader-with-variable-length-sequences-for-lstm-gru/
https://zhuanlan.zhihu.com/p/60129684
"""
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class PadSequence:
    """
    """
    def __call__(self, batch):
        # Let's assume that each element in "batch" is a list of tuple (data, label).
        # Sort the batch in the descending order
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        # Get each sequence and pad it
        sequences = [x[0] for x in sorted_batch]
        sequences_padded = pad_sequence(sequences)

        labels_padded = pad_sequence([x[1].long() for x in sorted_batch], padding_value=-1)
        return sequences_padded, labels_padded

class MyCustomDataset(Dataset):
    def __init__(self):
        file_ = "data/core_test.npz"
        with np.load(file_) as data:
            datalist = data['data']
            self.phone_to_idx = data['phone_to_idx']

        # translate the numpy data to tensor
        self.data = list()
        for d in datalist:
            self.data.append({
                'features': torch.from_numpy(d['features']),
                'phone_idx': torch.from_numpy(d['phone_idx'])
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
        return len(self.data)

if __name__ == "__main__":
    # make data set
    dataset = MyCustomDataset()
    dataloader = DataLoader(dataset=dataset, batch_size=10, shuffle=False, collate_fn=PadSequence())
    for features, labels in dataloader:
        print(features.shape)
        print(labels.shape)
