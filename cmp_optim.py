import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils import train
from utils import evalation
from utils.dataloader import TIMITDataset
from utils.dataloader import pad_seqs_to_batch
from models.bilstm import LSTMClassifier
import config

torch.manual_seed(100)

class Config(config.Config):
    part_labeled = 0.3
    optim = "adam"
    adam_betas = (0.9, 0.999)  # for adam
    steplr_gamma = 0.1
    steplr_stepsize = 10


if __name__ == "__main__":

    # configuration
    Config.optim = os.environ.get('optim', Config.optim)
    Config.part_labeled = float(os.environ.get('part_labeled', Config.part_labeled))
    print("========CONFIG===========")
    print("optim = ", Config.optim)
    print("part_labeled = ", Config.part_labeled)
    print("========CONFIG===========")

    #
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        cudnn.benchmark = True

    # ================================================================
    # data
    traindata = TIMITDataset(root="./data", split="train")
    n_labels = int(Config.part_labeled*len(traindata))
    n_unlables = len(traindata) - n_labels
    lbl_dataset, unlbl_dataset = torch.utils.data.random_split(
        traindata, [n_labels, n_unlables])

    #
    lbl_trainloader = DataLoader(dataset=lbl_dataset, batch_size=Config.batch_size,
                             shuffle=Config.shuffle, collate_fn=pad_seqs_to_batch)
    # ========================================================================
    validdata = TIMITDataset(root="./data", split="valid")
    validloader = DataLoader(dataset=validdata, batch_size=Config.batch_size,
                            collate_fn=pad_seqs_to_batch)
    testdata =  TIMITDataset(root="./data", split="test")
    testloader = DataLoader(dataset=testdata, batch_size=Config.batch_size,
                            collate_fn=pad_seqs_to_batch)
    # ========================================================================
    if n_labels < Config.batch_size:
        Config.batch_size = n_labels
    # ========================================================================
    # train the baseline network
    net = LSTMClassifier(
            n_feature=Config.n_features,
            n_class=Config.n_classes,
            n_hidden=Config.n_hidden_nodes, num_layers=3)

    # ==========================
    # set up optimizer
    # ==========================
    optimizer = optim.SGD(
            net.parameters(), lr=Config.init_lr, momentum=Config.momentum,
            weight_decay=Config.weight_decay)
    if Config.optim == "adam":
        optimizer = optim.Adam(net.parameters(),
            lr=Config.init_lr,
            betas=Config.adam_betas,
            weight_decay=Config.weight_decay)
        scheduler = None
    elif Config.optim == "stepLR":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=Config.steplr_stepsize,
            gamma=Config.steplr_gamma)
    else:
        step_size = 2*np.int(np.floor(n_labels/Config.batch_size))
        print("[cmpOptim] CyclicLR step size = ", step_size)
        print("[cmpOptim] batch_size = ", Config.batch_size)
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer, Config.eta_min, Config.eta_max, step_size_up=step_size)

    net.to(device)
    # Print out the configuration
    best_valid_acc = -np.inf
    for epoch in range(Config.n_epochs):
        # train the network
        train(lbl_trainloader, net, optimizer, scheduler=scheduler, device=device)
        # evaluate it
        valid_acc = evalation(validloader, net, device=device, tag="Valid")
        test_acc =  evalation(testloader, net, device=device, tag="Test")

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_teacher = net.state_dict()
            corresp_test_acc = test_acc
            saving_dict = {
                'epoch': epoch,
                'state_dict': best_teacher,
                'best_valid_acc': best_valid_acc,
                'corresp_test_acc': corresp_test_acc
            }
            torch.save(saving_dict,
                Config.cmp_tar_fmt.format(
                    plbl=Config.part_labeled,
                    optim=Config.optim))
    # ========================================================================
    print("Finish training Baseline!")
    print(
        "[Result][Baseline] Best epoch : {epoch} \t"
        "Valid. Acc. : {vacc:.4f}% \t Test Acc.: {tacc:.4f}%".format(
            epoch=saving_dict['epoch'],
            vacc=best_valid_acc*100,
            tacc=corresp_test_acc*100))
