<p align="center"><img width=500 src="/docs/images/torchbench.png"></p>

Easily benchmark PyTorch models on selected tasks and datasets. 

## Installation

Requires Python 3.6+. 

```bash
pip install torchbench
```

## Usage

This library can be used together with the [sotabench](https://sotabench.com) website, or standalone. Below we'll describe the usage with the sotabench website. 

Steps to benchmark your model on the sotabench website:

1) Create a `sotabench.py` in the root of your repository. Below you can see an example `sotabench.py` file added to the [torchvision](https://github.com/pytorch/vision/tree/master/torchvision) repository to test one of its constituent models:

```python
from torchbench.image_classification import ImageNet
from torchvision.models.resnet import resnext101_32x8d
import torchvision.transforms as transforms
import PIL

# Define the transforms need to convert ImageNet data to expected model input
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

2) Run it locally on your machine to verify it works:

```bash
$ python sotabench.py
```

In addition, you can check your parameters (such as model names and arxiv IDs) by running:
```bash
$ sb check
```

Alternatively you can run the same logic within a Notebook if that is your preferred workflow.

3) Login and connect your repository to [sotabench](https://sotabench.com/add-model). After you connect your repository the website will re-evaluate your model on every commit of the `master` branch, to ensure the model is working and results are up-to-date - including if you add additional models to the benchmark file.  

You can also use the library without the sotabench website, by simply omitting step 3. In that case you also don't need to put in the paper details into the `benchmark()` method. 

## Benchmarks

### Image Classification on ImageNet

Image Classification on ImageNet benchmark is implemented in the [image_classification.ImageNet](https://github.com/paperswithcode/torchbench/blob/master/torchbench/image_classification/imagenet.py) class.

#### Benchmarking Pipeline

1. The model is put into evaluation mode and sent to the device
2. The ImageNet validation dataset is loaded and transformed using `input_transform`
3. The dataset is put into a DataLoader with options `batch_size` and `num_workers`
4. The model and dataset are passed into an evaluation function for the task, along with an optional `model_output_transform` function that can transform the outputs after inference
5. The transformed output is compared to expected output and Top 1 and Top 5 accuracy are calculated

#### Expected Inputs/Outputs

- Model `output` (following `model.forward()` and optionally `model_output_transform`) should be a 2D `torch.Tensor` containing the model output; first dimension should be output for each example (length `batch_size`) and second dimension should be output for each class in ImageNet (length 1000).

### Object Detection on COCO

Object Detection on the COCO benchmark is implemented in the [object_detection.COCO](https://github.com/paperswithcode/torchbench/blob/master/torchbench/object_detection/coco.py) class. 

#### Benchmarking Pipeline

1. The model is put into evaluation mode and sent to the device
2. The COCO dataset is loaded. 

Without `transforms`, COCO dataset returns a tuple for an index where:
- The first entry is a `PIL.Image`
- The second entry is a labels dictionary, with keys `'boxes', 'labels', 'masks', 'image_id', 'area', 'iscrowd'`, containing the labels.

With [default transforms](https://github.com/paperswithcode/torchbench/blob/master/torchbench/object_detection/coco.py), COCO dataset returns a tuple for an index where:
- The first entry is a `torch.tensor` (representing the image)
- The second entry is a labels dictionary, with keys `'boxes', 'labels', 'masks', 'image_id', 'area', 'iscrowd'`, where the data is of type `torch.tensor` rather than lists.

You can specify your own `transforms` to transform the data so your model can process it correctly.

3. The dataset is put into a DataLoader with options `batch_size` and `num_workers`, and collated using [coco_collate_fn](https://github.com/paperswithcode/torchbench/blob/master/torchbench/object_detection/coco.py). Alternatively you can pass in your own collate function. 
4. The model and dataset are passed into an evaluation function for the task, along with `model_output_transform` function that can transform the outputs and targets after inference. The default [model_output_transform](https://github.com/paperswithcode/torchbench/blob/master/torchbench/object_detection/coco.py) follows that used in the torchvision examples, but you can provide your own `model_output_transform` to get your model output in the right format.

The expected output is a list of dictionaries (length = batch_size), where each dictionary contains keys for `'boxes', 'labels', 'scores', 'masks'`, and each value is of the `torch.tensor` type.

5. The (transformed) model output is then converted to a dictionary with keys as the image ids, and values as a dictionary with the predictions (boxes, labels, scores, ...)

```result = {tar["image_id"].item(): out for tar, out in zip(target, output)}```
            
6. The result is passed into a COCO Evaluation pipeline and the results (Mean Average Precision) are calculated

### More benchmarks coming soon... 

## Contributing

All contributions welcome!



