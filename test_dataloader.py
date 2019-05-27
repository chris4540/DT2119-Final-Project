from utils.dataloader import pad_seqs_to_batch
from utils.dataloader import TIMITDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # make data set
    print("==============================================")
    dataset = TIMITDataset(root="./data", split="test")
    dataloader = DataLoader(dataset=dataset, batch_size=100, shuffle=False, collate_fn=pad_seqs_to_batch)
    for features, labels in dataloader:
        print(features.shape)
        print(labels.shape)

    print("==============================================")
    dataset = TIMITDataset(root="./data", split="valid")
    dataloader = DataLoader(dataset=dataset, batch_size=100, shuffle=False, collate_fn=pad_seqs_to_batch)
    for features, labels in dataloader:
        print(features.shape)
        print(labels.shape)
    print("==============================================")
    dataset = TIMITDataset(root="./data", split="train")
    dataloader = DataLoader(dataset=dataset, batch_size=100, shuffle=False, collate_fn=pad_seqs_to_batch)
    for features, labels in dataloader:
        print(features.shape)
        print(labels.shape)