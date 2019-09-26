# PASCAL VOC 2012

![VOC Dataset Examples](img/pascalvoc2012.png)

You can view the PASCAL VOC 2012 leaderboard [here](https://sotabench.com/benchmarks/semantic-segmentation-on-pascal-voc-2012).

!!! Warning
    Semantic Segmentations APIs in PyTorch are not very standardised across repositories, meaning that
    it may require a lot of glue to get them working with this evaluation procedure (which is based on torchvision).
    
    **For easier VOC integration with sotabench it is recommended to use the more general API [sotabencheval](https://paperswithcode.github.io/sotabench-eval/).**

## Getting Started

You'll need the following in the root of your repository:

- `sotabench.py` file - contains benchmarking logic; the server will run this on each commit
- `requirements.txt` file - Python dependencies to be installed before running `sotabench.py`
- `sotabench_setup.sh` *(optional)* - any advanced dependencies or setup, e.g. compilation

Once you connect your repository to [sotabench.com](https://www.sotabench.com), the platform 
will run your `sotabench.py` file whenever you commit to master. 

We now show how to write the `sotabench.py` file to evaluate a PyTorch object model with
the torchbench library, and to allow your results to be recorded and reported for the community.

## The VOC Evaluation Class

You can import the evaluation class from the following module:

``` python
from torchbench.semantic_segmentation import PASCALVOC
```

The `PASCALVOC` class contains several components used in the evaluation, such as the `dataset`:

``` python
PASCALVOC.dataset
# torchvision.datasets.voc.VOCSegmentation
```

And some default arguments used for evaluation (which can be overridden):

``` python
PASCALVOC.normalize
# <torchbench.semantic_segmentation.transforms.Normalize at 0x7f9d645d2160>

PASCALVOC.transforms
# <torchbench.semantic_segmentation.transforms.Compose at 0x7f9d645d2278>

PASCALVOC.send_data_to_device
# <function torchbench.utils.default_data_to_device>

PASCALVOC.collate_fn
# <function torchbench.semantic_segmentation.utils.default_seg_collate_fn>

PASCALVOC.model_output_transform
# <function torchbench.semantic_segmentation.utils.default_seg_output_transform>
```

We will explain these different options shortly and how you can manipulate them to get the
evaluation logic to play nicely with your model.

An evaluation call - which performs evaluation, and if on the sotabench.com server, saves the results - 
looks like the following through the `benchmark()` method:

``` python
from torchvision.models.segmentation import fcn_resnet101
model = fcn_resnet101(num_classes=21, pretrained=True)

PASCALVOC.benchmark(model=model,
    paper_model_name='FCN ResNet-101',
    paper_arxiv_id='1605.06211')
```

These are the key arguments: the `model` which is a usually a `nn.Module` type object, but more generally,
is any method with a `forward` method that takes in input data and outputs predictions.
`paper_model_name` refers to the name of the model and `paper_arxiv_id` (optionally) refers to 
the paper from which the model originated. If these two arguments match a record paper result,
then sotabench.com will match your model with the paper and compare your code's results with the
reported results in the paper.


## A full `sotabench.py` example

Below shows an example for the [torchvision](https://github.com/pytorch/vision/tree/master/torchvision) 
repository benchmarking a FCN ResNet-101 model:


``` python
from torchbench.semantic_segmentation import PASCALVOC
from torchbench.semantic_segmentation.transforms import (
    Normalize,
    Resize,
    ToTensor,
    Compose,
)
from torchvision.models.segmentation import fcn_resnet101
import torchvision.transforms as transforms
import PIL

def model_output_function(output, labels):
    return output['out'].argmax(1).flatten(), target.flatten()

def seg_collate_fn(batch):
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets

model = fcn_resnet101(num_classes=21, pretrained=True)

normalize = Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
my_transforms = Compose([Resize((520, 480)), ToTensor(), normalize])

PASCALVOC.benchmark(batch_size=32,
    model=model,
    transforms=my_transforms,
    model_output_transform=model_output_function,
    collate_fn=seg_collate_fn,
    paper_model_name='FCN ResNet-101',
    paper_arxiv_id='1605.06211')
```

## `PASCALVOC.benchmark()` Arguments

The source code for the PASCALVOC evaluation method can be found [here](https://github.com/paperswithcode/torchbench/blob/develop/torchbench/semantic_segmentation/pascalvoc.py).
We now explain each argument.

### model

**a PyTorch module, (e.g. a ``nn.Module`` object), that takes in VOC data and outputs detections.**

For example, from the torchvision repository:

``` python
from torchvision.models.segmentation import fcn_resnet101
model = fcn_resnet101(num_classes=21, pretrained=True)
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
input_transform = transforms.Compose([
    transforms.Resize(512, PIL.Image.BICUBIC),
    transforms.ToTensor(),
])
```

### target_transform

**Composing the transforms used to transform the target data**

### transforms

**Composing the transforms used to transform the input data (the images) and the target data (the labels) 
in a dual fashion - for example resizing the pair of data jointly.** 

Below shows an example; note the
fact that the `__call__` takes in two arguments and returns two arguments (ordinary `torchvision` transforms
return one result).

``` python
from torchvision.transforms import functional as F

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class ImageResize(object):
    def __init__(self, resize_shape):
        self.resize_shape = resize_shape

    def __call__(self, image, target):
        image = F.resize(image, self.resize_shape)
        return image, target
        
transforms = Compose([ImageResize((512, 512)), ToTensor()])
```

Note that the default transforms are:

``` python
from torchbench.semantic_segmentation.transforms import (Normalize, Resize, ToTensor, Compose)
normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transforms = Compose([Resize((520, 480)), ToTensor(), normalize])
```

### model_output_transform

**(callable, optional): An optional function
                that takes in model output (after being passed through your
                ``model`` forward pass) and transforms it. Afterwards, the
                output will be passed into an evaluation function.**
                
The model output transform is a function that you can pass in to transform the model output
after the data has been passed into the model. This is useful if you have to do further 
processing steps after inference to get the predictions in the right format for evaluation.

The model evaluation for each batch is as follows from [utils.py](https://github.com/paperswithcode/torchbench/blob/db9fbdf5567350b8316336ca4f3fd27a04999347/torchbench/semantic_segmentation/utils.py) 
are:

``` python
with torch.no_grad():
    for i, (input, target) in enumerate(iterator):
        input, target = send_data_to_device(input, target, device=device)
        output = model(input)
        output, target = model_output_transform(output, target)
        confmat.update(target, output)
```

The default `model_output_transform` is:

``` python
def default_seg_output_transform(output, target):
    return output["out"].argmax(1).flatten(), target.flatten()
```

We can see the `output` and `target` are flattened to 1D tensors, and in the case of the output,
we take the maximum predicted class to compare against for accuracy. Each element in each tensor
represents a pixel, and contains a class, e.g. class 6, and we compare pixel-by-pixel the model
predictions against the ground truth labels to calculate the accuracy.

### collate_fn

**How the dataset is collated - an optional callable passed into the DataLoader**

As an example the default collate function is:

``` python
def default_seg_collate_fn(batch):
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets
```

### send_data_to_device

**An optional function specifying how the model is sent to a device**

As an example the PASCAL VOC default is:

``` python

def default_data_to_device(input, target=None, device: str = "cuda", non_blocking: bool = True):
    """Sends data output from a PyTorch Dataloader to the device."""

    input = input.to(device=device, non_blocking=non_blocking)

    if target is not None:
        target = target.to(device=device, non_blocking=non_blocking)

    return input, target
```

### data_root

**data_root (str): The location of the VOC dataset - change this
                parameter when evaluating locally if your VOC data is
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
    paper. See the VOC benchmark page for model names,
    https://sotabench.com/benchmarks/semantic-segmentation-on-pascal-voc-2012, e.g. on the paper
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
    are metric names, values are metric values. e.g::**

``` python
{'Accuracy': 0.745, 'Mean IOU': 0.592}.
```

Ensure that the metric names match those on the sotabench
leaderboard - for VOC it should be 'Accuracy', 'Mean IOU'.

### pytorch_hub_url

**pytorch_hub_url (str, optional): Optional linking to PyTorch Hub
    url if your model is linked there; e.g:
    'nvidia_deeplearningexamples_waveglow'.**

## Need More Help?

Head on over to the [Computer Vision](https://forum.sotabench.com/c/cv) section of the sotabench
forums if you have any questions or difficulties.
