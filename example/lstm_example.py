"""
https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
https://github.com/L1aoXingyu/pytorch-beginner/blob/master/05-Recurrent%20Neural%20Network/recurrent_network.py
https://stackoverflow.com/questions/53455780/pytorch-lstm-target-dimension-in-calculating-cross-entropy-loss
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
data = [
    # vec, class
    ([1, 2, 3], 0),
    ([2, 3, 4], 1),
    ([3, 4, 5], 2),
    ([4, 5, 6], 0),
    ([5, 6, 7], 1),
    ([6, 7, 8], 2),
]
tmp = list()
for d in data:
    _, y = d
    tmp.append(y)
num_classes = len(set(tmp))
# ==========================================================
# Test input
# n_hidden = 1
# n_targets = 1
# model = nn.LSTM(input_size=3, hidden_size=n_hidden, num_layers=1)
# # initialize the cell status and the hidden input
# input_ = torch.randn(4, 1, 3)  # (n_seq, n_batch, input_size)
# hidden_state = torch.randn(1, 1, n_hidden)
# cell_state = torch.randn(1, 1, n_hidden)
# h, c = hidden_state, cell_state
# out, (h, c) = model(input_, (h, c))
# ==========================================================
# build the input from data
n_feature = 3
n_seq = len(data)
input_ = torch.empty((n_seq, 1, n_feature))
targets = torch.empty((n_seq), dtype=torch.long)
for t, d in enumerate(data):
    x, y = d
    input_[t, 0, :] = torch.as_tensor(x)
    targets[t] = torch.as_tensor(y)

# ======================================================
class LSTMToy(nn.Module):
    """
    Wrapper of true lstm model
    """
    def __init__(self, n_feature, n_class, n_hidden=2):
        super().__init__()
        self.n_feature = n_feature
        self.n_hidden = n_hidden
        self.lstm = nn.LSTM(input_size=n_feature, hidden_size=n_hidden, num_layers=1)
        self.classifier = nn.Linear(n_hidden, n_class)

    def forward(self, x):
        h = torch.randn(1, 1, self.n_hidden)
        s = torch.randn(1, 1, self.n_hidden)
        # out, (h, s) = self.lstm(x, (h, s))
        out, (_, _)= self.lstm(x, (h, s))
        out = out[:, -1, :]  # flatten the axis=1
        out = self.classifier(out)
        return out
# ===============================================================
net = LSTMToy(3, num_classes)
loss_func = nn.CrossEntropyLoss()
# try eval
with torch.no_grad():
    preds = net(input_)
    eval_scores = loss_func(preds, targets)
    print(eval_scores)

# optimizer = optim.SGD(net.parameters(), lr=0.1)
optimizer = optim.Adam(net.parameters(), lr=0.1)
# best_loss = np.inf
for epoch in range(700):
    net.zero_grad()
    preds = net(input_)
    loss = loss_func(preds, targets)
    loss.backward()
    optimizer.step()
    loss = loss.float().item()

print("Final loss = ", loss)

with torch.no_grad():
    preds = net(input_)
    values, indices = torch.max(preds, 1)
    print(indices)