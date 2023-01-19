from torch.utils.data import Dataset
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import torch

from typing import List
import csv

class SignLanguageMNIST(Dataset):
    """Sign Language classification dataset.

    Utility for loading Sign Language dataset into PyTorch. Dataset posted on
    Kaggle in 2017, by an unnamed author with username `tecperson`:
    https://www.kaggle.com/datamunge/sign-language-mnist

    Each sample is 1 x 1 x 28 x 28, and each label is a scalar.
    """

    @staticmethod
    def get_label_mapping():
        """
        We map all labels to [0, 23]
        ignoring j and z due to time domain
        """
        ##ignore z
        mapping = list(range(25)) 
        ##ignore j
        mapping.pop(9)
        return mapping

    @staticmethod
    def read_label_samples_from_csv(path: str):
        """
        Assumes first column is label.
        Next 784 values are pixels to represent 28x28 image
        """

        mapping = SignLanguageMNIST.get_label_mapping()
        labels,samples = [],[]
        with open(path) as f:
            _ = next(f) #skip headers
            for line in csv.reader(f):
                label = int(line[0])
                labels.append(mapping.index(label))
                ##appends 784 pixels as an int list
                samples.append(list(map(int,line[1:])))
            return labels,samples

    def __init__(self,
            path: str="data/sign_mnist_train.csv",
            mean: List[float]=[0.485],
            std: List[float]=[0.229]):
        
        labels,samples = SignLanguageMNIST.read_label_samples_from_csv(path)
        self._samples = np.array(samples, dtype=np.uint8).reshape((-1,28,28,1))
        self._labels = np.array(labels, dtype=np.uint8).reshape((-1,1))

        self._mean = mean
        self._std = std


    def __len__(self):
        return len(self._labels)

    def __getitem__(self,idx):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(28, scale=(0.8,1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self._mean, std= self._std)])

        return {
                'image' : transform(self._samples[idx]).float(),
                'label': torch.from_numpy(self._labels[idx]).float()
                }

def get_train_test_loaders(batch_size=32):
    trainset = SignLanguageMNIST('data/sign_mnist_train.csv')
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size, shuffle=True)

    testset = SignLanguageMNIST('data/sign_mnist_train.csv')
    testloader = torch.utils.data.DataLoader(testset,batch_size=batch_size, shuffle=False)
    return trainloader, testloader

if __name__ == '__main__':
    loader,_ = get_train_test_loaders(2)
    print(next(iter(loader)))


