import os

import numpy as np
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random

class Cifar100_FL(Dataset):
    def __init__(self, data_path, superclass=[], subclass=[], percent=1.0, train=True, transform=None):
        if train:
            data_path = os.path.join(data_path, 'train')
        else:
            data_path = os.path.join(data_path, 'test')

        for i, super_class in enumerate(superclass):
            if i == 0:
                self.images, self.labels = np.load(
                    os.path.join(data_path, str(super_class)+"_"+str(subclass[i])+'.pkl'), allow_pickle=True)
            else:
                images, labels = np.load(os.path.join(data_path, str(super_class)+"_"+str(subclass[i])+'.pkl'),
                                             allow_pickle=True)
                self.images = np.concatenate([self.images, images], axis=0)
                self.labels = np.concatenate([self.labels, labels], axis=0)

        self.transform = transform
        self.labels = self.labels.astype(np.long).squeeze()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = Image.fromarray(image, mode='RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, label

def prepare_data_cifar100(args):
    transform_cifar = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    train_loaders = []
    test_loaders = []
    random.seed(args.seed)
    for i in range(args.num_clients):
        superclass = list(range(20))
        # subclass = [random.randint(0, 4) for _ in range(20)]
        subclass = [i]*20
        cifar_trainset = Cifar100_FL(data_path='dataset/cifar100', superclass=superclass, subclass=subclass,
                                     train=True, transform=transform_cifar)
        cifar_testset = Cifar100_FL(data_path='dataset/cifar100', superclass=superclass, subclass=subclass,
                                    train=False, transform=transform_cifar)
        cifar_train_loader = torch.utils.data.DataLoader(cifar_trainset, batch_size=args.batch_size, shuffle=True)
        cifar_test_loader = torch.utils.data.DataLoader(cifar_testset, batch_size=args.batch_size, shuffle=False)
        train_loaders.append(cifar_train_loader)
        test_loaders.append(cifar_test_loader)

    return train_loaders, test_loaders
