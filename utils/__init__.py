import numpy as np
import torch
import time
# from tqdm import tqdm
from torch.nn.utils.rnn import pad_packed_sequence

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

def train_one_epoch(train_loader, model, optimizer, device="cuda"):
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

    # print statistics
    train_loss = train_loss / len(train_loader)
    acc = 100 * correct / total
    used_time = time.time() - start_time
    print('Train Time used: %d \t Loss: %.3f | Train Acc: %.3f%% (%d/%d)' %
        (used_time, train_loss, acc, correct, total))