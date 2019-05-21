"""
Ref:
https://gist.github.com/williamFalcon/f27c7b90e34b4ba88ced042d9ef33edd
"""
import torch
import torch.nn as nn
import numpy as np
from torch.nn import utils as nn_utils
from torch.nn.functional import log_softmax
from torch import LongTensor
import torch.optim as optim

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i+n]

class LSTMToy(nn.Module):
    """
    Wrapper of true lstm model
    """
    def __init__(self, n_feature, n_class, n_hidden=3):
        super().__init__()
        self.n_feature = n_feature
        self.n_hidden = n_hidden
        self.lstm = nn.LSTM(input_size=n_feature, hidden_size=n_hidden, num_layers=1)
        self.classifier = nn.Linear(n_hidden, n_class)

    def forward(self, x):
        out, (_, _)= self.lstm(x)
        out, _ = nn_utils.rnn.pad_packed_sequence(out)
        out = self.classifier(out)
        return out

    def get_loss(self, logits, targets, pad_fill_val=-1):
        """
        Get the loss term with autograd
        Args:
            logits with padding, the output from the network
            logits.shape = <time, batch, class>
            Targets: padded targets, <time, batch>
            filled values = -1
        """
        log_p = log_softmax(logits, dim=-1) # take log_softmax over last dim

        # flatten <time, batch> => <time*batch>
        Y_mat = targets.view(-1)  # collospe all axis
        mask = (Y_mat != pad_fill_val)
        n_batch = log_p.size(1)
        n_time = log_p.size(0)
        log_p_flatten = log_p.view(-1, log_p.size(-1))

        extra_loss = log_p_flatten[LongTensor(range(n_time*n_batch)), Y_mat]
        losses = torch.masked_select(extra_loss, mask)

        cross_entropy_loss = -torch.sum(losses) / n_batch
        return cross_entropy_loss


if __name__ == "__main__":
    batch_size = 100
    max_length = 600
    n_features = 13
    n_classes = 61
    traindata = np.load('data/traindata_thin.npz')['traindata']
    net = LSTMToy(n_features, n_classes)
    # ======================================================
    optimizer = optim.Adam(net.parameters(), lr=0.05)
    for epoch in range(5):
        for data in chunks(traindata, batch_size):
            # obtain batch data
            batch_X = torch.zeros((max_length, batch_size, n_features))
            batch_Y = torch.ones((max_length, batch_size), dtype=torch.long) * -1
            seq_lengths = []

            # transform the data
            max_lengths = 0
            for i, d in enumerate(data):
                seq_len = d['lmfcc'].shape[0]
                seq_lengths.append(seq_len)
                batch_X[:seq_len, i, :] = torch.from_numpy(d['lmfcc'])
                batch_Y[:seq_len, i] = torch.from_numpy(np.array(d['targets']))
                if seq_len > max_lengths:
                    max_lengths = seq_len
            # ============================================================
            pack_X = nn_utils.rnn.pack_padded_sequence(
                batch_X, seq_lengths, enforce_sorted=False)
            batch_Y = batch_Y[:max_lengths,:]
            #
            out = net(pack_X)
            loss = net.get_loss(out, batch_Y)
            loss.backward()
            optimizer.step()
            loss = loss.float().item()
            print("Loss = ", loss)
