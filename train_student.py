import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.nn.functional import softmax
from torch.nn.functional import log_softmax
import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.utils.data import DataLoader
from utils import train
from utils import evalation
from utils import train_with_teacher_model
from utils.dataloader import TIMITDataset
from utils.dataloader import pad_seqs_to_batch
from models.lstm import LSTMClassifier
from models.bilstm import BiLSTMClassifier
from config import Config


if __name__ == "__main__":

    # configuration
    # obtain part_labeled
    Config.part_labeled = float(os.environ.get('part_labeled', Config.part_labeled))
    Config.temp = float(os.environ.get('temp', Config.temp))
    print("========CONFIG===========")
    print("part_labeled = ", Config.part_labeled)
    print("temp = ", Config.temp)
    print("========CONFIG===========")

    #
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        cudnn.benchmark = True

    # ================================================================
    # data
    traindata = TIMITDataset(root="./data", split="train")
    trainloader = DataLoader(dataset=traindata, batch_size=Config.batch_size,
                             shuffle=Config.shuffle, collate_fn=pad_seqs_to_batch)
    # ========================================================================
    validdata = TIMITDataset(root="./data", split="valid")
    validloader = DataLoader(dataset=validdata, batch_size=Config.batch_size,
                            collate_fn=pad_seqs_to_batch)
    testdata =  TIMITDataset(root="./data", split="test")
    testloader = DataLoader(dataset=testdata, batch_size=Config.batch_size,
                            collate_fn=pad_seqs_to_batch)
    # ========================================================================
    # teacher
    teacher = BiLSTMClassifier(
            n_feature=Config.n_features,
            n_class=Config.n_classes,
            n_hidden=Config.n_hidden_nodes, num_layers=3)
    teacher_save = torch.load(Config.teacher_tar_fmt.format(plbl=Config.part_labeled))
    teacher.load_state_dict(teacher_save['state_dict'])
    teacher.to(device)
    # ========================================================================
    # student
    student = LSTMClassifier(
            n_feature=Config.n_features,
            n_class=Config.n_classes,
            n_hidden=Config.n_hidden_nodes, num_layers=3)
    student.to(device)

    # make optimizer and scheduler
    step_size = 2*np.int(np.floor(len(traindata)/Config.batch_size))
    print("[Student] CyclicLR step size = ", step_size)
    optimizer = optim.SGD(
        student.parameters(), lr=Config.init_lr, momentum=Config.momentum,
        weight_decay=Config.weight_decay)
    scheduler = optim.lr_scheduler.CyclicLR(
        optimizer, Config.eta_min, Config.eta_max, step_size_up=step_size)

    best_valid_acc = -np.inf
    for epoch in range(Config.n_epochs):
        # train with all data (use only the input)
        train_with_teacher_model(
            trainloader, student, teacher, Config.temp,
            optimizer, scheduler=scheduler, device=device)

        # evaluate it
        valid_acc = evalation(validloader, student, device=device, tag="Valid")
        test_acc =  evalation(testloader, student, device=device, tag="Test")

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_student = student.state_dict()
            corresp_test_acc = test_acc

            saving_dict = {
                'epoch': epoch+1,
                'state_dict': best_student,
                'best_valid_acc': best_valid_acc,
                'corresp_test_acc': corresp_test_acc
            }

            torch.save(saving_dict,
                'student_plbl{plbl}_T{temp}.chkpt.tar'.format(
                    plbl=Config.part_labeled,
                    temp=Config.temp))
    print("Finish training student!")
    print(
        "[Result][Student] Valid. Acc. : {vacc:.4f}% \t Test Acc.: {tacc:.4f}%".format(
            vacc=best_valid_acc*100, tacc=corresp_test_acc*100))