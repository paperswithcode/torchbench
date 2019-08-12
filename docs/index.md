# Welcome to the Torchbench Documentation!

You have reached the docs for the **torchbench** library. Torchbench allows you to easily 
benchmark PyTorch models on selected tasks and datasets. It can be used in conjunction with the 
[sotabench](http://www.sotabench.com) website to record results for models, so the community
can compare model performance on different tasks.

## Getting Started : Benchmarking on ImageNet

The core structure of sotabench is benchmark datasets organized by task. Below we'll describe
the usage with the sotabench website, utilising an example on ImageNet.

**Step One : Create a benchmark.py file in the root of your repository**

Below you can see an example benchmark.py file added to the *torchvision* repository to test one of its models:

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

**Step Two : Run locally to verify that it works** 

    python benchmark.py

You can also run the logic in a Jupyter Notebook if that is your preferred workflow.

**Step Three : Login and connect your repository to [sotabench](http://www.sotabench.com)**

After you connect your repository the website will re-evaluate your model on every commit, to ensure the model is working and results are up-to-date - including if you add additional models to the benchmark file.

You can also use the library without the [sotabench](http://www.sotabench.com) website, by simply omitting step 3. In that case you also don't need to put in the paper details into the ```benchmark()``` method.

## Installation

The library requires Python 3.6+. You can install via pip:

    pip install torchbench

## Contents

```eval_rst
.. toctree::
   :maxdepth: 2

   01.evaluating-multiple-models.md
   02.adding-paper-results.md
   03.model-naming-convention.md
   api/index.md
```


## Indices and tables

```eval_rst
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
```

## Support

If you get stuck you can head to our [Discourse](http://forum.sotabench.com) forum where you ask
questions on how to use the project. You can also find ideas for contributions,
and work with others on exciting projects.