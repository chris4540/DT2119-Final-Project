import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataloader import TIMITDataset
from utils.dataloader import pad_seqs_to_batch
from models.lstm import LSTMClassifier
import torch.backends.cudnn as cudnn
from utils import train_one_epoch
import numpy as np

#
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    cudnn.benchmark = True

if __name__ == "__main__":
    batch_size = 100
    net = LSTMClassifier(39, 48, n_hidden=78, num_layers=3)
    traindata = TIMITDataset(root="./data", split="train")
    trainloader = DataLoader(dataset=traindata, batch_size=batch_size,
                             shuffle=True, collate_fn=pad_seqs_to_batch)
    # ==============================
    # optimizer = optim.Adam(net.parameters(), lr=0.05)
    step_size = 2*np.int(np.floor(len(traindata)/batch_size))
    print(step_size)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, 1e-5, 1e-2, step_size_up=step_size)
    net.to(device)
    # net.train()
    for epoch in range(30):
        train_one_epoch(trainloader, net, optimizer, scheduler=scheduler, device=device)
