import torch
import torch.nn as nn
import numpy as np
from torch import LongTensor
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.functional import log_softmax


class LSTMClassifier(nn.Module):
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
        out, (_, _)= self.lstm(x)
        # unpack the lstm output for next steps
        out, _ = pad_packed_sequence(out)
        out = self.classifier(out)
        return out

    def get_loss(self, logits, packed_targets, pad_fill_val=-1):
        """
        Get the loss term with autograd
        Args:
            logits (Tensor): with padding, the output from the network
                logits.shape = <time, batch, class>
            packed_targets (PackedSequence): padded targets, <time, batch>

        """
        targets, _ = pad_packed_sequence(packed_targets, padding_value=-1)
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

     # def to(self, device):
     #     super().to(device)
     #     self.lstm.to(device)


