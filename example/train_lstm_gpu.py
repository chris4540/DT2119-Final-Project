"""
Copyright (c) 2019 WeiHong Sung <weihongs@kth.se>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.
"""
"""
Ref:
https://gist.github.com/williamFalcon/f27c7b90e34b4ba88ced042d9ef33edd
TODO:
https://www.codefull.net/2018/11/use-pytorchs-dataloader-with-variable-length-sequences-for-lstm-gru/
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

class LSTM(nn.Module):
    """
    Wrapper of true lstm model
    """
    def __init__(self, n_feature, n_class, n_hidden=3, num_layers=1, mode = 'lstm'):
        super().__init__()
        self.n_feature = n_feature
        self.hidden_size = n_hidden
        self.num_layers = num_layers
        self.mode = mode
        if self.mode == 'bi-lstm':
            self.lstm = nn.LSTM(input_size=n_feature, hidden_size=n_hidden, num_layers=num_layers, bidirectional=True)
            self.classifier = nn.Linear(2 * n_hidden, n_class)
        elif self.mode == 'lstm':
            self.lstm = nn.LSTM(input_size=n_feature, hidden_size=n_hidden, num_layers=num_layers)
            self.classifier = nn.Linear(n_hidden, n_class)
        else:
            print("Wrong mode number!")
            exit()



    def forward(self, x):
        # batch_size = x.batch_sizes[0]
        # hidden, state = self._get_init_hidden_states(batch_size)
        # out, (_, _)= self.lstm(x, (hidden, state))
        out, (_, _)= self.lstm(x)
        out, _ = nn_utils.rnn.pad_packed_sequence(out)
        out = self.classifier(out)
        return out

    def _get_init_hidden_states(self, batch_size):
        sizes = (self.num_layers, batch_size, self.hidden_size)
        h0 = torch.randn(*sizes)
        s0 = torch.randn(*sizes)
        return h0, s0

    def get_loss(self, logits, targets, pad_fill_val=-1):
        """
        Get the loss term with autograd
        Args:
            logits with padding, the output from the network
            logits.shape = <time, batch, class>
            Targets: padded targets, <time, batch>
            filled values = -1
        """
        logits = logits.contiguous()
        targets = targets.contiguous()
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
    # Use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 100
    max_length = 600
    n_features = 13
    max_epoch = 30
    n_classes = 21  # reduced from tri-phone to phoneme
    traindata = np.load('data/traindata_thin.npz')['traindata']
    net = LSTM(n_features, n_classes, n_hidden=30, mode='bi-lstm', num_layers=2).to(device)

    # ======================================================
    optimizer = optim.Adam(net.parameters(), lr=0.05)
    net.train()
    for epoch in range(max_epoch):
        train_loss = 0
        batch_cnt = 0
        total = 0
        correct = 0
        for data in chunks(traindata, batch_size):
            # the size of the batch. At the end we would like remaining
            n_data = len(data)
            # obtain batch data
            batch_X = torch.zeros((max_length, n_data, n_features)).to(device)
            batch_Y = torch.ones((max_length, n_data), dtype=torch.long) * -1
            batch_Y.to(device)
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
            pack_X = nn_utils.rnn.pack_padded_sequence(batch_X, seq_lengths, enforce_sorted=False)
            # seq_lengths.sort(reverse=True)
            # pack_X = nn_utils.rnn.pack_padded_sequence(batch_X, seq_lengths).to(device)

            # print(pack_X.batch_sizes[0])
            targets = batch_Y[:max_lengths, :n_data].to(device)

            # reset grad
            optimizer.zero_grad()
            # feed forward
            out = net(pack_X)
            loss = net.get_loss(out, targets)
            # back-prop
            # compute gradient and do SGD step
            loss.backward()
            optimizer.step()

            # compute statistics
            out = out.float()
            loss = loss.float()

            train_loss += loss.item()
            _, predicted = out.max(-1)

            # calculate mask and select from it
            mask = (targets != -1)
            predicted = torch.masked_select(predicted, mask)
            true_labels = torch.masked_select(targets, mask)

            total += true_labels.size(0)
            correct += predicted.eq(true_labels).sum().item()
            batch_cnt += 1

        train_loss = train_loss / batch_cnt
        acc = 100 * correct / total
        print("Epoch:", epoch, " Train_loss", train_loss, " Accuracy:", acc)
