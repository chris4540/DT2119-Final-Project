import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataloader import TIMITDataset
from utils.dataloader import pad_seqs_to_batch
from models.lstm import LSTMClassifier
import torch.backends.cudnn as cudnn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    cudnn.benchmark = True

if __name__ == "__main__":
    net = LSTMClassifier(39, 48, n_hidden=60)
    traindata = TIMITDataset(root="./data", split="train")
    trainloader = DataLoader(dataset=traindata, batch_size=100, shuffle=False, collate_fn=pad_seqs_to_batch)
    # ==============================
    optimizer = optim.Adam(net.parameters(), lr=0.05)
    net.to(device)
    net.train()
    for epoch in range(3):
        train_loss = 0
        batch_cnt = 0
        start_time = time.time()
        for pack_inputs, pack_targets in trainloader:
            pack_inputs = pack_inputs.to(device)
            pack_targets = pack_targets.to(device)

            # reset grad
            optimizer.zero_grad()

            # forward
            out = net(pack_inputs)
            loss = net.get_loss(out, pack_targets)
            # back-prop
            # compute gradient and do SGD step
            loss.backward()
            optimizer.step()

            # compute statistics
            loss = loss.float()
            train_loss += loss.item()
            batch_cnt += 1

        end_time = time.time()
        train_loss = train_loss / batch_cnt
        print(end_time - start_time, " ", train_loss)
