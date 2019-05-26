from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as trns
from scipy.io import loadmat
from PIL import Image
import os


class TIMITDataset(Dataset):
    def __init__(self, root, split):
        """
        Initialize paths, transforms, and so on
        """
        pass
        # self.transform = transform

        # # load image path and annotations
        # mat = loadmat(os.path.join(root, split+'_list.mat'), squeeze_me=True)
        # self.imgs = mat['file_list']
        # self.imgs = [os.path.join(root, 'Images', i) for i in self.imgs]
        # self.lbls = mat['labels']
        # assert len(self.imgs) == len(self.lbls), 'mismatched length!'
        # print ('Total data in {} split: {}'.format(split, len(self.imgs)))

        # # label from 0 to (len-1)
        # self.lbls = self.lbls - 1

    def __getitem__(self, index):
        ##############################################
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        ##############################################
        pass
        # imgpath = self.imgs[index]
        # img = Image.open(imgpath).convert('RGB')
        # lbl = int(self.lbls[index])
        # if self.transform is not None:
        #     img = self.transform(img)
        # return img, lbl

    def __len__(self):
        ##############################################
        ### Indicate the total size of the dataset
        ##############################################
        pass
        # return len(self.imgs)

if __name__ == "__main__":
    pass
# # create train/val datasets
# trainset = dogDataset(root='./dataset/dogsDataset',
#                       split='train',
#                       transform=train_transform)
# valset = dogDataset(root='./dataset/dogsDataset',
#                     split='test',
#                     transform=val_transform)

# # create train/val loaders
# train_loader = DataLoader(dataset=trainset,
#                           batch_size=16,
#                           shuffle=True,
#                           num_workers=4)
# val_loader = DataLoader(dataset=valset,
#                         batch_size=16,
#                         shuffle=False,
#                         num_workers=4)

# # Get images and labels in a mini-batch of train_loader
# for imgs, lbls in train_loader:
#     print ('Size of image:', imgs.size())  # batch_size*3*224*224
#     print ('Type of image:', imgs.dtype)   # float32
#     print ('Size of label:', lbls.size())  # batch_size
#     print ('Type of label:', lbls.dtype)   # int64(long)
#     break