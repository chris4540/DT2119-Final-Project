import numpy as np
import pandas as pd
import torch
import time
# from tqdm import tqdm
from torch.nn.utils.rnn import pad_packed_sequence
from torch.optim.lr_scheduler import CyclicLR
from torch.optim.lr_scheduler import StepLR

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
        mapping = TrainPhnToEvalPhnMapping(mapping_tsv=....)
        mapping = TrainPhnToEvalPhnMapping()

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

def evalation(data_loader, model, device='cuda'):
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
        for inputs, targets in tqdm(data_loader, desc="Evaluating"):
            # load them to GPU
            inputs = inputs.to(device)
            targets = targets.to(device)
            if device == 'cuda':
                inputs = inputs.half()

            # predict
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # calculate the correct classfied
    score = correct / total
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

    for pack_inputs, pack_targets in train_loader:
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