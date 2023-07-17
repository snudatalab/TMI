from .data_from_imagefolder import *
from .data_from_torchvision import *


dataloader_dict = {
    'caltech101': get_dataloaders,
    'caltech256': get_dataloaders,
    'cifar10': get_cifar10_dataloaders,
    'cifar100': get_cifar100_dataloaders,
    'fashionmnist': get_fashionmnist_dataloaders,
    'mnist': get_mnist_dataloaders,
    'svhn': get_svhn_dataloaders,
    'flowerphotos': get_dataloaders,
    'eurosat': get_dataloaders,
    'chest': get_dataloaders,
    'visda': get_dataloaders,
    'aircraft': get_dataloaders,
    'birds': get_dataloaders,
    'cars': get_dataloaders,
    'dtd': get_dataloaders,
    'food': get_dataloaders,
    'pet': get_dataloaders,
}

num_classes = {
    'caltech101': 102,
    'caltech256': 257,
    'cifar10': 10,
    'cifar100': 100,
    'fashionmnist': 10,
    'mnist': 10,
    'svhn': 10,
    'flowerphotos': 5,
    'eurosat': 10,
    'chest': 2,
    'visda': 12,
    'aircraft': 100,
    'birds': 200,
    'cars': 196,
    'dtd': 47,
    'food': 101,
    'pet': 37,
    'imagenet': 1000,
}
