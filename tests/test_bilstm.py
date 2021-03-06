import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataloader import TIMITDataset
from utils.dataloader import pad_seqs_to_batch
from models.bilstm import BiLSTMClassifier
import torch.backends.cudnn as cudnn
from utils import train
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    cudnn.benchmark = True

if __name__ == "__main__":
    net = BiLSTMClassifier(39, 48, n_hidden=78, num_layers=3)
    traindata = TIMITDataset(root="./data", split="train")
    trainloader = DataLoader(dataset=traindata, batch_size=100,
                             shuffle=True, collate_fn=pad_seqs_to_batch)
    # ==============================
    optimizer = optim.Adam(net.parameters(), lr=0.05)
    net.to(device)
    # net.train()
    for epoch in range(30):
        train(trainloader, net, optimizer, device=device)
