from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.dataloader.utils import get_transforms


def get_cifar10_dataloaders(data_name: str, input_size: int = 32, batch_size: int = 10, num_workers: int = 8):
    """
    Get CIFAR10 dataloaders
    Args:
        data_name: dataset name
        input_size: input dimension
        batch_size: batch size
        num_workers: number of workers

    Returns: train loader, test loader

    """
    transform = get_transforms(data_name=data_name, crop_size=input_size)

    train_set = datasets.CIFAR10(root='../data', download=True, train=True, transform=transform['train'])
    test_set = datasets.CIFAR10(root='../data', download=True, train=False, transform=transform['test'])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def get_cifar100_dataloaders(data_name: str, input_size: int, batch_size: int = 128, num_workers: int = 8):
    """
    Get CIFAR100 dataloaders
    Args:
        data_name: dataset name
        input_size: input dimension
        batch_size: batch size
        num_workers: number of workers

    Returns: train loader, test loader

    """
    transform = get_transforms(data_name=data_name, crop_size=input_size)

    train_set = datasets.CIFAR100(root='../data', download=True, train=True, transform=transform['train'])
    test_set = datasets.CIFAR100(root='../data', download=True, train=False, transform=transform['test'])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def get_fashionmnist_dataloaders(data_name: str, input_size: int = 224, batch_size: int = 128, num_workers: int = 8):
    """
    Get FashionMNIST dataloaders
    Args:
        data_name: dataset name
        input_size: input dimension
        batch_size: batch size
        num_workers: number of workers

    Returns: train loader, test loader

    """
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize((0.3481,), (0.3472,)),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.3481,), (0.3472,)),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ])

    train_set = datasets.FashionMNIST(root='../data', download=True, train=True, transform=train_transform)
    test_set = datasets.FashionMNIST(root='../data', download=True, train=False, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def get_mnist_dataloaders(data_name: str, input_size: int = 224, batch_size: int = 128, num_workers: int = 8):
    """
    Get MNIST dataloaders
    Args:
        data_name: dataset name
        input_size: input dimension
        batch_size: batch size
        num_workers: number of workers

    Returns: train loader, test loader

    """
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(input_size),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize((0.1703,), (0.3198,)),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.1703,), (0.3198,)),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ])

    train_set = datasets.MNIST(root='../data', download=True, train=True, transform=train_transform)
    test_set = datasets.MNIST(root='../data', download=True, train=False, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def get_svhn_dataloaders(data_name: str, input_size: int = 224, batch_size: int = 128, num_workers: int = 8):
    """
    Get SVHN dataloaders
    Args:
        data_name: dataset name
        input_size: input dimension
        batch_size: batch size
        num_workers: number of workers

    Returns: train loader, test loader

    """
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(input_size),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize((0.4364, 0.4435, 0.4748), (0.1951, 0.198, 0.1947)),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4364, 0.4435, 0.4748), (0.1951, 0.198, 0.1947)),
    ])

    train_set = datasets.SVHN(root='../data', download=True, split='train', transform=train_transform)
    test_set = datasets.SVHN(root='../data', download=True, split='test', transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
