# torchbench

Easily benchmark PyTorch models on selected tasks and datasets. 

## Installation

Requires Python 3.6+. 

```bash
pip install torchbench
```

## Usage

This library can be used together with the [sotabench](https://sotabench.com) website, or standalone. Below we'll describe the usage with the sotabench website. 

Steps to benchmark your model on the sotabench website:

1) Create a `benchmark.py` in the root of your repository. Below you can see an example `benchmark.py` file added to the [torchvision](https://github.com/pytorch/vision/tree/master/torchvision) repository to test one of its constituent models:

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
python benchmark.py
```

Alternatively you can run the same logic within a Notebook if that is your preferred workflow.

3) Login and connect your repository to [sotabench](https://sotabench.com/add-model). After you connect your repository the website will re-evaluate your model on every commit, to ensure the model is working and results are up-to-date - including if you add additional models to the benchmark file.  

You can also use the library without the sotabench website, by simply omitting step 3. In that case you also don't need to put in the paper details into the `benchmark()` method. 

## Benchmarks

### Image Classification on ImageNet

Image Classification on ImageNet benchmark is implemented in the [torchbench.image_classification.ImageNet](https://github.com/paperswithcode/torchbench/blob/master/torchbench/image_classification/imagenet.py) class. The implementation is using the [torchvision ImageNet dataset](https://pytorch.org/docs/stable/torchvision/datasets.html#imagenet). 

#### Benchmarking Pipeline

1. The model is put into evaluation mode and sent to the device
2. The ImageNet validation dataset is loaded and transformed using `input_transform`
3. The dataset is put into a DataLoader with options `batch_size` and `num_workers`
4. The model and dataset are passed into an evaluation function for the task, along with an optional `model_output_transform` function that can transform the outputs after inference.
5. The transformed output is compared to expected output and Top 1 and Top 5 accuracy are calculated

Once the benchmarking is complete, the results are printed to the screen (and when run on sotabench.com automatically stored in the database). 

#### Expected Inputs/Outputs

- Model `output` (following `model.forward()` and optionally `model_output_transform`) should be a 2D `torch.Tensor` containing the model output; first dimension should be output for each example (length `batch_size`) and second dimension should be output for each class in ImageNet (length 1000).

### More benchmarks coming soon... 

## Contributing

All contributions welcome!



