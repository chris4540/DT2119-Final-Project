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

class LSTMToy(nn.Module):
    """
    Wrapper of true lstm model
    """
    def __init__(self, n_feature, n_class, n_hidden=3, num_layers=1):
        super().__init__()
        self.n_feature = n_feature
        self.hidden_size = n_hidden
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=n_feature, hidden_size=n_hidden, num_layers=num_layers)
        self.classifier = nn.Linear(n_hidden, n_class)

    def forward(self, x):
        # batch_size = x.size(1)
        # hidden, state = self._get_init_hidden_states()
        out, (_, _)= self.lstm(x)
        out, _ = nn_utils.rnn.pad_packed_sequence(out)
        out = self.classifier(out)
        return out

    def _get_init_hidden_states(self, batch_size):
        sizes = (self.hparams.nb_lstm_layers, batch_size, self.hidden_size)
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
    batch_size = 100
    max_length = 600
    n_features = 13
    n_classes = 21  # reduced from tri-phone to phoneme
    traindata = np.load('data/traindata_thin.npz')['traindata']
    net = LSTMToy(n_features, n_classes)
    # ======================================================
    optimizer = optim.Adam(net.parameters(), lr=0.05)
    net.train()
    for epoch in range(30):
        train_loss = 0
        batch_cnt = 0
        total = 0
        correct = 0
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
            targets = batch_Y[:max_lengths,:i+1]
            #
            out = net(pack_X)
            loss = net.get_loss(out, targets)
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute statistics
            out = out.float()
            loss = loss.float()

            train_loss += loss.item()
            _, predicted = out.max(-1)

            mask = (targets != -1)
            predicted = torch.masked_select(predicted, mask)
            true_labels = torch.masked_select(targets, mask)

            total += true_labels.size(0)
            correct += predicted.eq(true_labels).sum().item()
            batch_cnt += 1


        train_loss = train_loss / batch_cnt
        acc = 100 * correct / total
        print(train_loss, acc)

