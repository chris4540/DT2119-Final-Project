from .lstm import LSTMClassifier
import torch.nn as nn

class BiLSTMClassifier(LSTMClassifier):
    def __init__(self, n_feature, n_class, n_hidden=3, num_layers=1):
        super().__init__(n_feature, n_class, n_hidden, num_layers)
        self.lstm = nn.LSTM(input_size=n_feature, hidden_size=n_hidden,
                            num_layers=num_layers, bidirectional=True)
        self.classifier = nn.Linear(2*n_hidden, n_class)