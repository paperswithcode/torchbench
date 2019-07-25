# torchbench

Easily benchmark PyTorch models on selected tasks and datasets.

*Work in progress version*

## Installation

Requires Python 3.6+. 

```bash
pip install git+ssh://github.com/paperswithcode/torchbench#egg=torchbench
```

## Example Usage

To benchmark a model create a `benchmark.py` file. For example, to benchmark the EfficientNet model on image classification on ImageNet:

```python
from torchbench.image_classification import ImageNet
import torch
import torchvision.transforms as transforms
import PIL

# load the model
model = torch.hub.load('rwightman/gen-efficientnet-pytorch', 
                       'efficientnet_b0', 
                       pretrained=True)

# transform ImageNet data into the format the model takes
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
input_transform = transforms.Compose([
    transforms.Resize(256, PIL.Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

# Run evaluation
ImageNet.benchmark(
    model=model,
    input_transform=input_transform
)

```

You can run the evaluation by running the file:

```bash
python benchmark.py
```

### Comparing to a paper

In the example above we've just evaluated the EfficientNet implementation, but we can also directly compare it to the results reported in the paper. 

To compare to a paper, add the names of the model and paper into the benchmark call:

```python
# Run evaluation and compare to paper
ImageNet.benchmark(
    model=model,
    input_transform=input_transform,
    paper_model_name='EfficientNet-B0',
    paper_arxiv_id='1905.11946'
)
```

We provide a free server to evaluate against paper results. Put your code into github, and then submit your github
repository to [sotabench.com](https://sotabench.com) and it will be automatically built, record the results and compare 
them to the paper. 

## TODO docs

[work in progress docs](docs/).

- Tutorial: step-by-step with exploring benchmarks and transforms 
- API reference manual
- Settings and env variables, e.g. to capture all output in JSON:

    ```bash
    export SOTABENCH_STORE_FILENAME='evaluation.json'
    ```

