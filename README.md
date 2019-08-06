# torchbench

Easily benchmark PyTorch models on selected tasks and datasets. 

## Installation

Requires Python 3.6+. 

```bash
pip install git+ssh://github.com/paperswithcode/torchbench#egg=torchbench
```

## Usage

This library can be used together with the [sotabench](https://sotabench.com) website, or standalone. Below we'll describe the usage with the sotabench website. 

Steps to benchmark your model on the sotabench website:

1) Create a `benchmark.py` in the root of your repository. Below you can see the example `benchmark.py` file added to the torchvision library to test one of the models there:

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

2) Run it locally on your machine to verify it works. 

```bash
python benchmark.py
```

3) Login and connect your repository to [sotabench](https://sotabench.com/add-model). After you connect your repository the website will re-evaluate your model on every commit, to ensure the model is working and results are up-to-date.  

You can also use the library without the sotabench website, by simply ommitting step 3. In that case you also don't need to put in the paper details into the `benchmark()` method. 

## Benchmarks

### Image Classification on ImageNet

Image Classification on ImageNet benchmark is implemented in the [torchbench.image_classification.ImageNet](https://github.com/paperswithcode/torchbench/blob/master/torchbench/image_classification/imagenet.py) class. The implementation is using the [torchvision ImageNet dataset](https://pytorch.org/docs/stable/torchvision/datasets.html#imagenet). 

The benchmarking pipeline is as follows:

1. The ImageNet validation dataset is loaded, if it doesn't exist locally it wil be downloaded
2. Data is split into batches and transformed using the specified (or default) `input_transform`
3. Transformed data is passed into the model as the only parameter
4. Model output is recorded and if any `output_transform` are specified they are applied to the output
5. The transformed output is compared with the labels from ImageNet and Top 1 and Top 5 accuracy calculated

Once the benchmarking is complete, the results are printed to the screen (and when run on sotabench.com automatically stored in the database). 

#### More benchmarks coming soon... 

## Contributing

All contributions welcome!



