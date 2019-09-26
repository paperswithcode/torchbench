# ImageNet

![ImageNet Dataset Examples](img/imagenet.jpeg)

You can view the ImageNet leaderboard [here](https://sotabench.com/benchmarks/image-classification-on-imagenet).

## Getting Started

You'll need the following in the root of your repository:

- `sotabench.py` file - contains benchmarking logic; the server will run this on each commit
- `requirements.txt` file - Python dependencies to be installed before running `sotabench.py`
- `sotabench_setup.sh` *(optional)* - any advanced dependencies or setup, e.g. compilation

Once you connect your repository to [sotabench.com](https://www.sotabench.com), the platform 
will run your `sotabench.py` file whenever you commit to master. 

We now show how to write the `sotabench.py` file to evaluate a PyTorch object model with
the torchbench library, and to allow your results to be recorded and reported for the community.
 
## The ImageNet Evaluation Class

You can import the evaluation class from the following module:

``` python
from torchbench.image_classification import ImageNet
```

The `ImageNet` class contains several components used in the evaluation, such as the `dataset`:

``` python
ImageNet.dataset
# torchvision.datasets.ImageNet
```

And some default arguments used for evaluation (which can be overridden):

``` python
ImageNet.normalize
# Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

ImageNet.input_transform
# Compose(
#   Resize(size=256, interpolation=PIL.Image.BILINEAR)
#   CenterCrop(size=(224, 224))
#   ToTensor()
#   Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# )

ImageNet.send_data_to_device
# <function torchbench.utils.default_data_to_device>
```

We will explain these different options shortly and how you can manipulate them to get the
evaluation logic to play nicely with your model.

An evaluation call - which performs evaluation, and if on the sotabench.com server, saves the results - 
looks like the following through the `benchmark()` method:

``` python
from torchvision.models.resnet import resnext101_32x8d

ImageNet.benchmark(
    model=resnext101_32x8d(pretrained=True),
    paper_model_name='ResNeXt-101-32x8d',
    paper_arxiv_id='1611.05431'
)
```

These are the key arguments: the `model` which is a usually a `nn.Module` type object, but more generally,
is any method with a `forward` method that takes in input data and outputs predictions.
`paper_model_name` refers to the name of the model and `paper_arxiv_id` (optionally) refers to 
the paper from which the model originated. If these two arguments match a record paper result,
then sotabench.com will match your model with the paper and compare your code's results with the
reported results in the paper.

## A full `sotabench.py` example

Below shows an example for the [torchvision](https://github.com/pytorch/vision/tree/master/torchvision) 
repository benchmarking a ResNeXt-101-32x8d model:

``` python
from torchbench.image_classification import ImageNet
from torchvision.models.resnet import resnext101_32x8d
import torchvision.transforms as transforms
import PIL

# Define the transforms need to convert ImageNet data to expected
# model input
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
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
## `ImageNet.benchmark()` Arguments

The source code for the ImageNet evaluation method can be found [here](https://github.com/paperswithcode/torchbench/blob/develop/torchbench/image_classification/imagenet.py).
We now explain each argument.

### model

**a PyTorch module, (e.g. a ``nn.Module`` object), that takes in ImageNet data and outputs detections.**

For example, from the torchvision repository:

``` python
from torchvision.models.resnet import resnext101_32x8d
model = resnext101_32x8d(pretrained=True)
```

### model_description

**(str, optional): Optional model description.**

For example:

``` python
model_description = 'Using ported TensorFlow weights'
```

### input_transform

**Composing the transforms used to transform the input data (the images), e.g. 
resizing (e.g ``transforms.Resize``), center cropping, to tensor transformations and normalization.**

For example:

``` python
import torchvision.transforms as transforms
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
input_transform = transforms.Compose([
    transforms.Resize(256, PIL.Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])
```

### target_transform

**Composing the transforms used to transform the target data**

### model_output_transform

**(callable, optional): An optional function
                that takes in model output (after being passed through your
                ``model`` forward pass) and transforms it. Afterwards, the
                output will be passed into an evaluation function.**
                
The model output transform is a function that you can pass in to transform the model output
after the data has been passed into the model. This is useful if you have to do further 
processing steps after inference to get the predictions in the right format for evaluation.

Most PyTorch models for Image Classification on ImageNet don't need to use this argument.

The model evaluation for each batch is as follows from [utils.py](https://github.com/paperswithcode/torchbench/blob/db9fbdf5567350b8316336ca4f3fd27a04999347/torchbench/image_classification/utils.py#L189) 
are:

``` python
with torch.no_grad():
    for i, (input, target) in enumerate(iterator):

        input, target = send_data_to_device(input, target, device=device)
        output = model(input)

        if model_output_transform is not None:
            output = model_output_transform(output, target, model=model)

        check_metric_inputs(output, target, test_loader.dataset, i)
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
```

Model output (following `model.forward()` and optionally `model_output_transform`) should be a 2D 
`torch.Tensor` containing the model output; first dimension should be output for each example 
(length batch_size) and second dimension should be output for each class in ImageNet (length 1000).

### send_data_to_device

**An optional function specifying how the model is sent to a device**

As an example the default is:

``` python

def default_data_to_device(input, target=None, device: str = "cuda", non_blocking: bool = True):
    """Sends data output from a PyTorch Dataloader to the device."""

    input = input.to(device=device, non_blocking=non_blocking)

    if target is not None:
        target = target.to(device=device, non_blocking=non_blocking)

    return input, target
```

### data_root

**data_root (str): The location of the ImageNet dataset - change this
                parameter when evaluating locally if your ImageNet data is
                located in a different folder (or alternatively if you want to
                download to an alternative location).**
                
Note that this parameter will be overriden when the evaluation is performed on the server,
so it is solely for your local use.

### num_workers

**num_workers (int): The number of workers to use for the DataLoader.**

### batch_size

**batch_size (int) : The batch_size to use for evaluation; if you get
                memory errors, then reduce this (half each time) until your
                model fits onto the GPU.**
                
### paper_model_name

**paper_model_name (str, optional): The name of the model from the
    paper - if you want to link your build to a machine learning
    paper. See the ImageNet benchmark page for model names,
    https://sotabench.com/benchmarks/image-classification-on-imagenet, e.g. on the paper
    leaderboard tab.**
    
### paper_arxiv_id
    
**paper_arxiv_id (str, optional): Optional linking to ArXiv if you
    want to link to papers on the leaderboard; put in the
    corresponding paper's ArXiv ID, e.g. '1611.05431'.**

### paper_pwc_id

**paper_pwc_id (str, optional): Optional linking to Papers With Code;
    put in the corresponding papers with code URL slug, e.g.
    'u-gat-it-unsupervised-generative-attentional'**
    
### paper_results

**paper_results (dict, optional) : If the paper you are reproducing
    does not have model results on sotabench.com, you can specify
    the paper results yourself through this argument, where keys
    are metric names, values are metric values. e.g:**

``` python
{'Top 1 Accuracy': 0.543, 'Top 5 Accuracy': 0.654}
```

Ensure that the metric names match those on the sotabench
leaderboard - for ImageNet it should be 'Top 1 Accuracy', 
'Top 5 Accuracy'

### pytorch_hub_url

**pytorch_hub_url (str, optional): Optional linking to PyTorch Hub
    url if your model is linked there; e.g:
    'nvidia_deeplearningexamples_waveglow'.**
 
## Need More Help?

Head on over to the [Computer Vision](https://forum.sotabench.com/c/cv) section of the sotabench
forums if you have any questions or difficulties.
