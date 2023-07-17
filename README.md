# Fast and Accurate Transferability Measurement by Evaluating Intra-class Feature Variance

This project is a Pytorch implementation of the paper Fast and Accurate Transferability Measurement by Evaluating Intra-class Feature Variance (TMI). 
TMI quantifies the degree of transferability observed between a pre-trained model and a target task, which is characterized by a target dataset.

## Prerequisites

Our implementation is based on Python 3.8 and Pytorch 1.12.0. The other dependencies are listed in `requirements.txt`.

## Datasets

We use 17 datasets in our experiments. The details of the datasets are listed in the following table.

| Datasets                                                                    | # of instances (train / test) | Category |
|-----------------------------------------------------------------------------|----------------------------------------|------------------|
| [Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02)                 | 7,315 / 1,829                          | Multiple domains |
| [Caltech-256](https://data.caltech.edu/records/nyy15-4j048)                 | 24,485 / 6,122                         | Multiple domains |
| [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)                     | 50,000 / 10,000                        | Multiple domains |
| [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)                    | 50,000 / 10,000                        | Multiple domains |
| [MNIST](http://yann.lecun.com/exdb/mnist/)                                  | 60,000 / 10,000                        | Digit            |
| [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist)            | 60,000 / 10,000                        | Fashion          |
| [SVHN](http://ufldl.stanford.edu/housenumbers/)                             | 73,257 / 26,032                        | Digit            |
| [FlowerPhotos](https://www.tensorflow.org/datasets/catalog/tf_flowers)	     | 2,936 / 734                            | Flower           |
| [EuroSAT](https://github.com/phelber/eurosat)                               | 21,600 / 5,400                         | Landscape        |
| [Chest X-Ray](https://data.mendeley.com/datasets/rscbjbr9sj/2)              | 5,216 / 623                            | Medicine         |
| [VisDA](http://ai.bu.edu/visda-2017/\#browse)                               | 152,397 / 72,372                       | Multiple domains |
| [FGVC-Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)       | 3,334 / 3,333                          | Aircraft         |
| [CUB-200](http://www.vision.caltech.edu/datasets/cub_200_2011/)             | 9,430 / 2,358                          | Animal           |
| [Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)              | 8,144 / 8,041                          | Auto             |
| [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/)                           | 4,512 / 1,128                          | Texture          |
| [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)     | 80,800 / 20,200                        | Food             |
| [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/)              | 3,680 / 3,669                          | Animal           |

## Usage

You can put features and labels into the TMI function to estimate a transferability score.

```python
from trf_measurement import TMI
# feature has shape of [N, D], label has shape [N,].
tmi = TMI(feature, label)
```

You can also run `main.py` to obtain both a transferability score and running time for measuring transferability.

```python
cd src
python main.py --src imagenet --dataset caltech101 --model resnet50 --batch_size 512
```

## Citation
Please cite this paper when you use our code.
```
@inproceedings{conf/iccv/XuK23,
  author    = {Huiwen Xu and
               U Kang},
  title     = {Fast and Accurate Transferability Measurement by Evaluating Intra-class Feature Variance},
  booktitle = {ICCV},
  year      = {2023},
}
```

## License
This software may be used only for non-commercial purposes (e.g., research eval>
Please contact Prof. U Kang (ukang@snu.ac.kr) if you want to use it for other p>

