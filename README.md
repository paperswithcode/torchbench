<p align="center"><img width=500 src="/docs/docs/img/torchbench.png"></p>

--------------------------------------------------------------------------------

[![PyPI version](https://badge.fury.io/py/torchbench.svg)](https://badge.fury.io/py/torchbench) [![Docs](https://img.shields.io/badge/Documentation-Here-<COLOR>.svg)](https://paperswithcode.github.io/torchbench/)

`torchbench` is a library that contains a collection of deep learning benchmarks you can use to benchmark your models, optimized for the PyTorch framework. It can be used in conjunction with the [sotabench](https://www.sotabench.com) service to record results for models, so the community can compare model performance on different tasks, as well as a continuous integration style service for your repository to benchmark your models on each commit.

## Benchmarks Supported

- [ImageNet](https://paperswithcode.github.io/torchbench/imagenet/) (Image Classification)
- [COCO](https://paperswithcode.github.io/torchbench/coco/) (Object Detection) - *partial support*
- [PASCAL VOC 2012](https://paperswithcode.github.io/torchbench/pascalvoc/) (Semantic Segmentation) - *partial support*

PRs welcome for further benchmarks! 

## Installation

Requires Python 3.6+. 

```bash
pip install torchbench
```

## Get Benching! 🏋️

You should read the [full documentation here](https://paperswithcode.github.io/torchbench/index.html), which contains guidance on getting started and connecting to [sotabench](https://www.sotabench.com).

The API is optimized for PyTorch implementations. For example, if you wanted to benchmark a [torchvision](https://github.com/pytorch/vision) model for ImageNet, you would write a `sotabench.py` file like this:

```python
from torchbench.image_classification import ImageNet
from torchvision.models.resnet import resnext101_32x8d
import torchvision.transforms as transforms
import PIL

# Define the transforms need to convert ImageNet data to expected model input
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225])
input_transform = transforms.Compose([
    transforms.Resize(256, PIL.Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

# Run the benchmark
ImageNet.benchmark(
    model=resnext101_32x8d(pretrained=True),
    paper_model_name='ResNeXt-101-32x8d',
    paper_arxiv_id='1611.05431',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1
)
```

Sotabench will run this on each commit and record the results. For other tasks, such as object detection and semantic segmentation, implementations are much less standardized than for image classification. It is therefore recommended you use [sotabencheval](https://github.com/paperswithcode/sotabench-eval/) for these tasks - although there are experimental benchmarks for [COCO](https://paperswithcode.github.io/torchbench/coco/) and [PASCAL VOC](https://paperswithcode.github.io/torchbench/pascalvoc/).

## Contributing

All contributions welcome!
