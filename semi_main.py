import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn.functional import softmax
from torch.nn.functional import log_softmax
import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.utils.data import DataLoader
from utils import train
from utils import evalation
from utils.dataloader import TIMITDataset
from utils.dataloader import pad_seqs_to_batch
from models.lstm import LSTMClassifier
from models.bilstm import BiLSTMClassifier

class Config:
    batch_size = 100
    n_epochs = 1
    init_lr = 0.01  # this would not take effect as using cyclic lr
    momentum = 0.9
    weight_decay = 5e-4
    eta_min = 1e-5
    eta_max = 1e-2
    shuffle = True
    n_hidden_nodes = 78
    part_labeled = 0.3  # the percentage of labeled data
    n_features = 39
    n_classes = 48
    temp = 1

def repad_batchout(batch_out, seq_lengths, padding_val=0):
    """
    Re-pad the batch output of a network
    Args:
        batch_out (Tensor): batch output of a network.
            Size = (max_seq_len, n_batch, n_classes)
        seq_lengths (list/LongTensor)

    Return:
        same as input, but repadded with padding_val
    """
    # re-pack it as the packed seqences
    pack = pack_padded_sequence(batch_out, seq_lengths)

    # de-pack it and fill zeros
    ret, _ = pad_packed_sequence(pack, padding_value=padding_val)
    return ret

def get_kd_loss(teacher_logits, student_logits, seq_lens):

    # get n_batchs
    n_batchs = student_logits.size(1)

    # repad
    student_logits = repad_batchout(student_logits, seq_lens)
    teacher_logits = repad_batchout(teacher_logits, seq_lens)

    sum_kd_loss = nn.KLDivLoss(reduction='sum')(
            log_softmax(student_logits/Config.temp, dim=-1),
            softmax(teacher_logits/Config.temp, dim=-1))
    loss = sum_kd_loss / n_batchs
    return loss


if __name__ == "__main__":
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
    print(len(lbl_dataset))
    print(len(unlbl_dataset))
    #
    lbl_trainloader = DataLoader(dataset=lbl_dataset, batch_size=Config.batch_size,
                             shuffle=Config.shuffle, collate_fn=pad_seqs_to_batch)
    unlbl_trainloader = DataLoader(dataset=unlbl_dataset, batch_size=Config.batch_size,
                             shuffle=Config.shuffle, collate_fn=pad_seqs_to_batch)

    # ========================================================================
    validdata = TIMITDataset(root="./data", split="valid")
    validloader = DataLoader(dataset=validdata, batch_size=Config.batch_size,
                            collate_fn=pad_seqs_to_batch)
    testdata =  TIMITDataset(root="./data", split="test")
    testloader = DataLoader(dataset=testdata, batch_size=Config.batch_size,
                            collate_fn=pad_seqs_to_batch)
    # ========================================================================


    # ========================================================================
    # train teacher
    teacher = BiLSTMClassifier(
            n_feature=Config.n_features,
            n_class=Config.n_classes,
            n_hidden=Config.n_hidden_nodes, num_layers=1)

    step_size = 2*np.int(np.floor(len(lbl_dataset)/Config.batch_size))
    optimizer = optim.SGD(
        teacher.parameters(), lr=Config.init_lr, momentum=Config.momentum,
        weight_decay=Config.weight_decay)
    scheduler = optim.lr_scheduler.CyclicLR(
        optimizer, Config.eta_min, Config.eta_max, step_size_up=step_size)
    teacher.to(device)
    # Print out the configuration
    print("The CyclicLR step size = ", step_size)
    best_valid_acc = -np.inf
    for epoch in range(Config.n_epochs):
        # train the network
        train(lbl_trainloader, teacher, optimizer, scheduler=scheduler, device=device)
        # evaluate it
        valid_acc = evalation(validloader, teacher, device=device, tag="Valid")
        test_acc =  evalation(testloader, teacher, device=device, tag="Test")

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_teacher = teacher.state_dict()
            corresp_test_acc = test_acc
    # ========================================================================
    print("Finish training teacher!")
    teacher.load_state_dict(best_teacher)
    print(
        "[Result][Teacher] Valid. Acc. : {vacc:.4f}% \t Test Acc.: {tacc:.4f}%".format(
            vacc=best_valid_acc*100, tacc=corresp_test_acc*100))
    # ========================================================================
    # TODO
    student = LSTMClassifier(
            n_feature=Config.n_features,
            n_class=Config.n_classes,
            n_hidden=Config.n_hidden_nodes, num_layers=1)
    student.to(device)
    optimizer = optim.SGD(
        teacher.parameters(), lr=Config.init_lr, momentum=Config.momentum,
        weight_decay=Config.weight_decay)
    scheduler = optim.lr_scheduler.CyclicLR(
        optimizer, Config.eta_min, Config.eta_max, step_size_up=step_size)
    from tqdm import tqdm
    from torch.optim.lr_scheduler import CyclicLR
    from torch.optim.lr_scheduler import StepLR
    student.train()
    for epoch in range(Config.n_epochs):
        train_loss = 0
        for pack_inputs, _ in tqdm(unlbl_trainloader, desc="Train"):
            pack_inputs = pack_inputs.to(device)

            _, seq_lens = pad_packed_sequence(pack_inputs)

            # compute output
            teacher.eval()
            with torch.no_grad():
                teacher_logits = teacher(pack_inputs)

            output = student(pack_inputs)
            loss = get_kd_loss(output, teacher_logits, seq_lens)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            train_loss += loss.item()
            _, predicted = output.max(-1)

            if isinstance(scheduler, CyclicLR):
                scheduler.step()
        if isinstance(scheduler, StepLR):
            scheduler.step()

        train_loss = train_loss / len(unlbl_trainloader)
        print(train_loss)

        # evaluate it
        valid_acc = evalation(validloader, student, device=device, tag="Valid")
        test_acc =  evalation(testloader, student, device=device, tag="Test")


