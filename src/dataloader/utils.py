from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

norm_dict = {
    'caltech101': ((0.5377, 0.5103, 0.4802), (0.3016, 0.2954, 0.3072)),
    'caltech256': ((0.5406, 0.5148, 0.483), (0.3052, 0.3012, 0.3137)),
    'cifar10': ((0.489, 0.4756, 0.4396), (0.2364, 0.2328, 0.25)),
    'cifar100': ((0.5052, 0.4782, 0.43), (0.2551, 0.243, 0.2622)),
    'fashionmnist': ((0.3481,), (0.3472,)),
    'mnist': ((0.1703,), (0.3198,)),
    'svhn': ((0.4364, 0.4435, 0.4748), (0.1951, 0.198, 0.1947)),
    'sprites': (1, 1),
    'flowerphotos': ((0.5059, 0.4435, 0.3129), (0.2927, 0.2644, 0.2892)),
    'eurosat': ((0.3446, 0.3808, 0.4086), (0.2022, 0.1355, 0.1136)),
    'chest': (0.5834, 0.1639),
    'visda': ((0.7971, 0.7946, 0.7892), (0.2455, 0.2497, 0.2594)),
    'aircraft': ((0.4619, 0.4883, 0.5080), (0.2412, 0.2378, 0.2653)),
    'birds': ((0.4576, 0.4680, 0.4040), (0.2463, 0.2434, 0.2699)),
    'cars': ((0.4280, 0.4147, 0.4125), (0.2983, 0.2956, 0.3021)),
    'dtd': ((0.5024, 0.4487, 0.4011), (0.2828, 0.2678, 0.2701)),
    'food': ((0.5254, 0.4187, 0.3120), (0.2823, 0.2752, 0.2702)),
    'pet': ((0.4541, 0.4205, 0.3734), (0.2756, 0.2671, 0.2695)),
}


def get_transforms(data_name: str, resize_size: int = 256, crop_size: int = 224):
    """
    Get the transforms for the data loaders
    Args:
        data_name: data name
        resize_size: resize dimension
        crop_size: crop dimension

    Returns: transform dictionary

    """
    transform = {}
    transform['train'] = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(*norm_dict[data_name])
    ])
    transform['test'] = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(*norm_dict[data_name])
    ])
    return transform


class ImageDataset(Dataset):
    def __init__(self, dataset, data_name, class_to_idx, transform=None):
        """
        Initialize
        Args:
            dataset: dataset include image and label
            data_name: data name
            class_to_idx: class to index dictionary
            transform: transform
        """
        super().__init__()
        self.dataset = dataset
        self.data_name = data_name
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.idx_to_class = {int(v): k for k, v in self.class_to_idx.items()}

    def __getitem__(self, index):
        """
        Get image and label by index
        Args:
            index: index

        Returns: image and label

        """
        img, label = self.dataset[index]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        """
        Get the length of the dataset
        Returns: number of instances

        """
        return len(self.dataset)


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class ImageList(Dataset):
    def __init__(self, image_path_list, data_name, class_to_idx, transform=None):
        """
        Initialize
        Args:
            image_path_list: list of image paths
            data_name: data name
            class_to_idx: class to index dictionary
            transform: transform
        """
        super().__init__()
        self.image_path_list = image_path_list
        self.data_name = data_name
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.idx_to_class = None if class_to_idx is None else {int(v): k for k, v in self.class_to_idx.items()}
        self.loader = rgb_loader

    def __getitem__(self, index):
        """
        Get image and label by index
        Args:
            index: index

        Returns: image and label

        """
        path, label = self.image_path_list[index]
        img = self.loader(path)
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        """
        Get the length of the dataset
        Returns: number of instances

        """
        return len(self.image_path_list)
