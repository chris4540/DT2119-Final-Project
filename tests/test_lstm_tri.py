"""
Baseline model
"""
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataloader import TIMITDataset
from utils.dataloader import pad_seqs_to_batch
from models.lstm import LSTMClassifier
import torch.backends.cudnn as cudnn
from utils import train
from utils import evalation
import numpy as np

class Config:
    batch_size = 100
    n_epochs = 30
    init_lr = 0.01  # this would not take effect as using cyclic lr
    momentum = 0.9
    weight_decay = 5e-4
    eta_min = 1e-5
    eta_max = 1e-2
    shuffle = True
    num_hidden_nodes = 78

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        cudnn.benchmark = True
    net = LSTMClassifier(n_feature=39, n_class=48,
            n_hidden=Config.num_hidden_nodes, num_layers=3)

    traindata = TIMITDataset(root="./data", split="train")
    trainloader = DataLoader(dataset=traindata, batch_size=Config.batch_size,
                             shuffle=Config.shuffle, collate_fn=pad_seqs_to_batch)
    validdata = TIMITDataset(root="./data", split="valid")
    validloader = DataLoader(dataset=validdata, batch_size=Config.batch_size,
                            collate_fn=pad_seqs_to_batch)
    testdata =  TIMITDataset(root="./data", split="test")
    testloader = DataLoader(dataset=testdata, batch_size=Config.batch_size,
                            collate_fn=pad_seqs_to_batch)
    # ========================================================================

    # ========================================================================
    step_size = 2*np.int(np.floor(len(traindata)/Config.batch_size))
    optimizer = optim.SGD(
        net.parameters(), lr=Config.init_lr, momentum=Config.momentum,
        weight_decay=Config.weight_decay)
    scheduler = optim.lr_scheduler.CyclicLR(
        optimizer, Config.eta_min, Config.eta_max, step_size_up=step_size)
    net.to(device)
    # Print out the configuration
    print("The CyclicLR step size = ", step_size)
    for epoch in range(Config.n_epochs):
        # train the network
        train(trainloader, net, optimizer, scheduler=scheduler, device=device)
        # evaluate it
        valid_acc = evalation(validloader, net, device=device, tag="Valid")
        test_acc =  evalation(testloader, net, device=device, tag="Test")

