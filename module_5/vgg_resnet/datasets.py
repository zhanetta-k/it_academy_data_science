import os
import sys

import torchvision
from torchvision import datasets
import torchvision.transforms
from torch.utils.data import DataLoader
import PIL.Image


batch_size = 16
class_list = ('ant', 'bee')
num_classes = 2
channels = 3

train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.RandomCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(os.path.join(sys.path[0],
                                  "hymenoptera_data/train"),
                                  transform=train_transforms)
test_data = datasets.ImageFolder(os.path.join(sys.path[0],
                                 "hymenoptera_data/val"),
                                 transform=test_transforms)

train_loader = DataLoader(dataset=train_data,
                          batch_size=batch_size,
                          drop_last=True,
                          shuffle=True)
test_loader = DataLoader(dataset=test_data,
                         batch_size=batch_size,
                         shuffle=False)


# Augmented train data
train_transforms_aug = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomRotation(degrees=30,
                                          interpolation=PIL.Image.BILINEAR),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

train_data_aug = datasets.ImageFolder(os.path.join(sys.path[0],
                                      "hymenoptera_data/train"),
                                      transform=train_transforms_aug)

train_loader_aug = DataLoader(dataset=train_data,
                              batch_size=batch_size,
                              drop_last=True,
                              shuffle=True)
