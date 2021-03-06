import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
from tqdm import tqdm
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import CyclicLR
from torch.optim.lr_scheduler import StepLR
from torch.nn.functional import softmax
from torch.nn.functional import log_softmax


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class TrainPhnToEvalPhnMapping(metaclass=Singleton):
    """
    Usage:
        create the mapping by one of the followings
        mapping = TrainPhnToEvalPhnMapping(mapping_tsv=....).get_dict()
        mapping = TrainPhnToEvalPhnMapping().get_dict()

        # use it
        eval_labels = np.vectorize(mapping.get)(train_labels)
        # or
        # labels is a torch.Tensor
        labels.map_(labels, mapping.get)
    """
    def __init__(self, mapping_tsv=None):
        """
        Args:
            the mapping tsv: could be data/map/phones.60-48-39.map
        """
        if mapping_tsv is None:
            print("Using the default mapping tsv")
            self.mapping_tsv = "data/map/phones.60-48-39.map"
        else:
            self.mapping_tsv = mapping_tsv
        # load the csv
        df = pd.read_csv(self.mapping_tsv, sep="\t", index_col=0)
        df = df.dropna()  # drop the q phone
        # ===================================================================
        # translate phone to index by simple dict comprehension
        train_phn_idx = {k: i for i, k in enumerate(df['train'].unique())}
        df['train_idx'] = df['train'].map(train_phn_idx)
        eval_phn_idx = {k: i for i, k in enumerate(df['eval'].unique())}
        df['eval_idx'] = df['eval'].map(eval_phn_idx)
        # ===================================================================
        # extract only the index column and translate them
        sub_df = df[['train_idx', 'eval_idx']]
        sub_df = sub_df.set_index('train_idx')

        # the mapping from phone index for training to evaluation
        self.map = sub_df['eval_idx'].to_dict()

    def get(self, key):
        return self.map[key]

    def get_dict(self):
        """
        Return a clone of self.map
        """
        return {k: v for k, v in self.map.items()}

def map_phone_to_idx(phone, phone_to_idx):
    """
    Args:
        phone (list[str]): list of labels
        phone_to_idx (dict): mapping from string lable to index
    Returns:
        list of phone index
    """
    ret = np.vectorize(phone_to_idx.get)(phone)
    return ret

def evalation(data_loader, model, device='cuda', tag=""):
    """
    Run evaluation
    Return:
        The accurancy
    """
    # switch to evaluate mode
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for pack_inputs, pack_targets in data_loader:
            # load them to device
            pack_inputs = pack_inputs.to(device)
            pack_targets = pack_targets.to(device)

            # predict
            outputs = model(pack_inputs)
            _, predicted = outputs.max(-1)

            # calculate mask and select from it
            padded_targets, _ = pad_packed_sequence(pack_targets, padding_value=-1)
            mask = (padded_targets != -1)
            predicted = torch.masked_select(predicted, mask)
            true_labels = torch.masked_select(padded_targets, mask)

            # translate them back to cpu
            predicted = predicted.cpu()
            true_labels = true_labels.cpu()

            # map the training labels to evaluation labels
            mapping = TrainPhnToEvalPhnMapping().get_dict()
            true_labels.map_(true_labels, mapping.get)
            predicted.map_(predicted, mapping.get)

            total += true_labels.size(0)
            correct += predicted.eq(true_labels).sum().item()

    # calculate the correct classfied
    score = correct / total
    acc = score*100.0
    print("[Eval] {tag} Acc. : {acc:.2f}% \t ({crr}/{tol})".format(
            acc=acc, crr=correct, tol=total, tag=tag))
    return score

def train(train_loader, model, optimizer, scheduler=None, device="cuda"):
    """
    Run one train epoch
    """
    # switch to train mode
    model.train()

    start_time = time.time()
    train_loss = 0
    total = 0
    correct = 0

    for pack_inputs, pack_targets in tqdm(train_loader, desc="Train"):
        pack_inputs = pack_inputs.to(device)
        pack_targets = pack_targets.to(device)

        # compute output
        output = model(pack_inputs)
        loss = model.get_loss(output, pack_targets)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        train_loss += loss.item()
        _, predicted = output.max(-1)

        # calculate mask and select from it
        padded_targets, _ = pad_packed_sequence(pack_targets, padding_value=-1)
        mask = (padded_targets != -1)
        predicted = torch.masked_select(predicted, mask)
        true_labels = torch.masked_select(padded_targets, mask)

        total += true_labels.size(0)
        correct += predicted.eq(true_labels).sum().item()
        if isinstance(scheduler, CyclicLR):
            scheduler.step()

    if isinstance(scheduler, StepLR):
        scheduler.step()

    # print statistics
    train_loss = train_loss / len(train_loader)
    acc = 100 * correct / total
    used_time = time.time() - start_time
    print('Train Time used: %d \t Loss: %.3f | Train Acc: %.3f%% (%d/%d)' %
        (used_time, train_loss, acc, correct, total))


def train_with_teacher_model(
        train_loader, model, teacher_model, temp, optimizer,
        scheduler=None, device="cuda"):
    """
    Train with a probabilistic logits from teacher model
    """

    # switch to train mode
    model.train()
    teacher_model.eval()

    start_time = time.time()
    train_loss = 0

    for pack_inputs, _ in tqdm(train_loader, desc="LogitsTrain"):
        _, seq_lens = pad_packed_sequence(pack_inputs)
        pack_inputs = pack_inputs.to(device)

        # compute teacher logits
        with torch.no_grad():
            target_logit = teacher_model(pack_inputs)

        # compute output
        output = model(pack_inputs)
        loss = get_kd_loss(output, target_logit, seq_lens, temp=temp)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.float()

        # measure accuracy and record loss
        train_loss += loss.item()

        if isinstance(scheduler, CyclicLR):
            scheduler.step()

    if isinstance(scheduler, StepLR):
        scheduler.step()

    # print statistics
    train_loss = train_loss / len(train_loader)
    used_time = time.time() - start_time
    print('LogitsTrain Time used: %d \t Loss: %.3f' % (used_time, train_loss))

def get_kd_loss(student_logits, teacher_logits, seq_lens, temp):

    # get n_batchs
    n_batchs = student_logits.size(1)

    # repad with zero s.t. two logtis has not different at those "masked"
    # time and batch => contribute no loss
    student_logits = repad_batchout(student_logits, seq_lens)
    teacher_logits = repad_batchout(teacher_logits, seq_lens)

    sum_kd_loss = nn.KLDivLoss(reduction='sum')(
            log_softmax(student_logits/temp, dim=-1),
            softmax(teacher_logits/temp, dim=-1))
    loss = (temp**2)*sum_kd_loss / n_batchs
    return loss

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


def train_with_teacher_logits(
        train_loader, model, target_logits, temp, optimizer,
        scheduler=None, device="cuda"):
    """
    Train with a probabilistic logits
    """

    # switch to train mode
    model.train()

    start_time = time.time()
    train_loss = 0

    i = 0
    for pack_inputs, _ in tqdm(train_loader, desc="Precal-LogitsTrain"):
        _, seq_lens = pad_packed_sequence(pack_inputs)
        pack_inputs = pack_inputs.to(device)

        # get logist
        target_logit = target_logits[i]

        # compute output
        output = model(pack_inputs)
        loss = get_kd_loss(output, target_logit, seq_lens, temp=temp)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.float()

        # measure accuracy and record loss
        train_loss += loss.item()

        i += 1
        if isinstance(scheduler, CyclicLR):
            scheduler.step()

    if isinstance(scheduler, StepLR):
        scheduler.step()

    # print statistics
    train_loss = train_loss / len(train_loader)
    used_time = time.time() - start_time
    print('LogitsTrain Time used: %d \t Loss: %.3f' % (used_time, train_loss))