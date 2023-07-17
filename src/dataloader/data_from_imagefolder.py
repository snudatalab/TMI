import scipy.io
import torchvision

from torch.utils.data import DataLoader, random_split
from src.dataloader.utils import get_transforms, ImageDataset, ImageList

path_dict = {
    'caltech101': 'caltech101/101_ObjectCategories/',
    'caltech256': 'caltech256/256_ObjectCategories/',
    'flowerphotos': 'flower_photos/',
    'eurosat': 'eurosat/2750/',
    'chest': 'chest_xray/',
    'visda': 'visDA/',
    'aircraft': 'airCraft/',
    'birds': 'CUB_200_2011/CUB_200_2011/images/',
    'cars': 'cars/',
    'dtd': 'dtd/images/',
    'food': 'food/images/',
    'pet': 'oxford-iiit-pet/',
}


def get_dataloaders(data_name: str, input_size: int = 224, batch_size: int = 128, num_workers: int = 8,
                    root_path: str = '../data/'):
    """
    Get dataloaders for imagefolder datasets
    Args:
        data_name: data name
        input_size: input dimension
        batch_size: batch size
        num_workers: number of workers
        root_path: root path of data

    Returns: train loader, test loader

    """

    if any(m == data_name for m in ('caltech101', 'caltech256', 'flowerphotos', 'eurosat', 'birds', 'dtd', 'food')):
        dataset = torchvision.datasets.ImageFolder(root=root_path + path_dict[data_name])
        class_to_idx = dataset.class_to_idx
        train_size = int(len(dataset) * 0.8)
        test_size = len(dataset) - train_size
        subset_train, subset_test = random_split(dataset, [train_size, test_size])
    elif data_name == 'chest':
        subset_train = torchvision.datasets.ImageFolder(root=root_path + path_dict[data_name] + 'train/')
        subset_test = torchvision.datasets.ImageFolder(root=root_path + path_dict[data_name] + 'test/')
        class_to_idx = subset_train.class_to_idx
    elif data_name == 'visda':
        subset_train = torchvision.datasets.ImageFolder(root=root_path + path_dict[data_name] + 'train/')
        class_to_idx = subset_train.class_to_idx
        _test_lines = open(root_path + path_dict[data_name] + 'image_list.txt').readlines()
        subset_test_path = list(
            map(lambda line: [root_path + path_dict[data_name] + 'test/' + line.strip().split(' ')[0],
                              int(line.strip().split(' ')[1])], _test_lines))
    elif data_name == 'aircraft':
        class_to_idx = None
        _train_lines = open(root_path + path_dict[data_name] + 'train.csv').readlines()[1:]
        subset_train_path = list(map(lambda line: [
            root_path + path_dict[data_name] + 'fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/images/' +
            line.strip().split(',')[0], int(line.strip().split(',')[2])], _train_lines))
        _test_lines = open(root_path + path_dict[data_name] + 'test.csv').readlines()[1:]
        subset_test_path = list(map(lambda line: [
            root_path + path_dict[data_name] + 'fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/images/' +
            line.strip().split(',')[0], int(line.strip().split(',')[2])], _test_lines))
    elif data_name == 'cars':
        class_to_idx = None
        _train_mat_file = scipy.io.loadmat(root_path + path_dict[data_name] + 'cars_train_annos.mat')
        subset_train_path = list(
            map(lambda line: [root_path + path_dict[data_name] + 'cars_train/' + 'cars_train/' + line[5][0],
                              line[4][0][0] - 1], _train_mat_file['annotations'][0]))
        _test_mat_file = scipy.io.loadmat(root_path + path_dict[data_name] + 'cars_test_annos_withlabels.mat')
        subset_test_path = list(
            map(lambda line: [root_path + path_dict[data_name] + 'cars_test/' + 'cars_test/' + line[5][0],
                              line[4][0][0] - 1], _test_mat_file['annotations'][0]))
    elif data_name == 'pet':
        class_to_idx = None
        _train_lines = open(root_path + path_dict[data_name] + 'annotations/trainval.txt').readlines()
        subset_train_path = list(
            map(lambda line: [root_path + path_dict[data_name] + 'images/' + line.strip().split(' ')[0] + '.jpg',
                              int(line.strip().split(' ')[1]) - 1], _train_lines))
        _test_lines = open(root_path + path_dict[data_name] + 'annotations/test.txt').readlines()
        subset_test_path = list(
            map(lambda line: [root_path + path_dict[data_name] + 'images/' + line.strip().split(' ')[0] + '.jpg',
                              int(line.strip().split(' ')[1]) - 1], _test_lines))
    else:
        raise NameError

    # Get dataset
    transform = get_transforms(data_name=data_name, crop_size=input_size)
    if any(m == data_name for m in ('aircraft', 'cars', 'pet')):
        dataset_train = ImageList(subset_train_path, data_name, class_to_idx, transform=transform['train'])
    else:
        dataset_train = ImageDataset(subset_train, data_name, class_to_idx, transform=transform['train'])
    if any(m == data_name for m in ('visda', 'aircraft', 'cars', 'pet')):
        dataset_test = ImageList(subset_test_path, data_name, class_to_idx, transform=transform['test'])
    else:
        dataset_test = ImageDataset(subset_test, data_name, class_to_idx, transform=transform['test'])

    # Get dataloader
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  drop_last=False)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 drop_last=False)

    return dataloader_train, dataloader_test
