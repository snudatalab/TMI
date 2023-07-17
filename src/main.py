import time
import copy
import argparse
import torch
import torch.nn as nn
import torchvision

from dataloader import dataloader_dict, num_classes
from models import pretrained_model_dict, weights_dict, pretrained_path_dict
from trf_measurement import TMI


def set_arguments():
    """
    Set arguments for the experiment
    Returns: arguments

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='imagenet', help='source dataset')
    parser.add_argument('--dataset', type=str, default='caltech101', help='target dataset',
                        choices=['caltech101', 'caltech256', 'cifar10', 'cifar100', 'mnist', 'fashionmnist', 'svhn',
                                 'flowerphotos', 'eurosat', 'chest', 'visda', 'aircraft', 'birds', 'cars', 'dtd',
                                 'food', 'pet'])
    parser.add_argument('--model', type=str, default='resnet50', help='model architecture')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    args = parser.parse_args()
    return args


def get_model(model_name, src, num_classes, device):
    """
    Get feature extractor from a pre-trained model
    Args:
        model_name: model name
        src: src dataset
        num_classes: number of classes
        device: device to eval the model

    Returns: feature extractor

    """
    # Load model
    if any(m in model_name for m in ('barlowtwins', 'moco', 'swav', 'simsiam', 'dino')):
        model = torchvision.models.resnet50(weights=None)

        checkpoint = torch.load(f'./pretrained_models/{model_name.split("_")[0]}/{pretrained_path_dict[model_name]}')
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        if 'swav' in model_name:
            # remove prefixe "module."
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            for k, v in model.state_dict().items():
                if k not in list(state_dict):
                    print('key "{}" could not be found in provided state dict'.format(k))
                elif state_dict[k].shape != v.shape:
                    print('key "{}" is of different shape in model and provided state dict'.format(k))
                    state_dict[k] = v
        elif 'moco' in model_name:
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
        elif 'simsiam' in model_name:
            for k in list(state_dict.keys()):
                # retain only encoder up to before the embedding layer
                if k.startswith('module.encoder') and not k.startswith('module.encoder.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
        elif 'dino' in model_name:
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    else:
        model = pretrained_model_dict[model_name](weights=weights_dict[model_name].IMAGENET1K_V1)

    if src != 'imagenet':
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes[src])
        checkpoint = torch.load(f'./save/source/{src}/model/{model_name}/best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded model: {model_name} trained on {src}')

    # Extract feature extractor
    if any(m in model_name for m in ('densenet', 'mobilenet', 'mnasnet', 'efficientnet')):
        model.classifier = nn.Identity()
    elif any(m in model_name for m in ('resnet', 'shufflenet', 'resnext', 'wide_resnet',
                                       'regnet', 'barlowtwins', 'swav', 'moco', 'dino', 'simsiam')):
        model.fc = nn.Identity()
    elif 'swin' in model_name:
        model.head = nn.Identity()
    elif 'vit' in model_name:
        model.heads = nn.Identity()
    else:
        raise NameError

    return model.to(device)


def trf_measure(output, k, num_workers):
    """
    Compute TMI
    Args:
        output: output of the model including features and ground truth
        k: number of nearest neighbors
        num_workers: number of workers

    Returns:

    """
    representation, ground_truth = list(map(lambda x: x.cpu().numpy(), output))

    s_time = time.time()
    trf = TMI(representation, ground_truth, k=k, workers=num_workers)
    trf_time = time.time() - s_time
    return trf, trf_time


def model_eval(model, dataloader, device):
    """
    Compute the representation of a model on a dataset
    Args:
        model: model to eval
        dataloader: dataloader
        device: device to eval the model

    Returns: representation and ground truth

    """
    representation = []
    ground_truth = []

    model.eval()

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)

            hidden_rep = model(x)
            hidden_rep = hidden_rep.reshape([hidden_rep.shape[0], -1])

            representation.append(hidden_rep.detach().cpu())
            ground_truth.append(y.detach())

    representation = torch.cat(representation, dim=0)
    ground_truth = torch.cat(ground_truth, dim=0)

    return [representation, ground_truth]


def main():
    """
    Compute the transferability of a model on a dataset
    """
    args = set_arguments()
    args.num_workers = 40 if torch.cuda.is_available() else 2
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'==> Prepare {args.dataset} and eval {args.model}')
    dataloader, _ = dataloader_dict[args.dataset](args.dataset, 224, args.batch_size, args.num_workers)
    models = get_model(args.model, args.src, num_classes, args.device)
    output = model_eval(models, dataloader, args.device)

    # Estimate the transferability
    results = trf_measure(output, args.k, args.num_workers)
    results = [str(f'{i:.2f}') for i in results]
    print(f'Transferability score: {results[0]} \nRunning time: {results[1]}s')


if __name__ == '__main__':
    main()
