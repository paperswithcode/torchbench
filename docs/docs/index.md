# Welcome to torchbench!

<img src="img/torchbench.png" width=500>

You have reached the docs for the [torchbench](https://github.com/paperswithcode/torchbench] library. This library contains a collection of deep learning benchmarks you can use to
benchmark your models, optimized for the PyTorch framework. It can be used in conjunction with the 
[sotabench.com](http://www.sotabench.com) website to record results for models, so the community
can compare model performance on different tasks, as well as a continuous integration style
service for your repository to benchmark your models on each commit.

**torchbench** is a framework-optimized library, meaning it is designed to take advantage of PyTorch based features
and standardisation. If this is too constraining, you can use alternative libraries that are framework-independent,
 e.g. [sotabencheval](https://paperswithcode.github.io/sotabench-eval/).

## Getting Started : Benchmarking on ImageNet

**Step One : Create a sotabench.py file in the root of your repository**

This contains a call to your model, metadata about your model, and options for evaluation such as dataset
processing logic and data loader logic such as the batch size. Below is an example for the [torchvision](https://github.com/pytorch/vision)
repository:

``` python
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

**Step Two : Run locally to verify that it works** 

```
python sotabench.py
```

You can also run the logic in a Jupyter Notebook if that is your preferred workflow.

To verify your benchmark will run and all parameters are correct you can use the included CLI checking tool:

```
$ sb check
```

**Step Three : Login and connect your repository to [sotabench](http://www.sotabench.com)**

Create an account on [sotabench](http://www.sotabench.com), then head to your user page. Click the
**Connect a GitHub repository** button:

<img width=400 src="img/connect.png">

Then follow the steps to connect the repositories that you wish to benchmark:

<img width=500 src="img/connect2.png">

After you connect your repository, the sotabench servers will re-evaluate your model on every commit, 
to ensure the model is working and results are up-to-date - including if you add additional models to the benchmark file.

## Installation

The library requires Python 3.6+. You can install via pip:

```
pip install torchbench
```

## Support

If you get stuck you can head to our [Discourse](http://forum.sotabench.com) forum where you ask
questions on how to use the project. You can also find ideas for contributions,
and work with others on exciting projects.