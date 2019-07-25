from setuptools import setup

PACKAGE_NAME = "torchbench"
LICENSE = "Apache 2.0"
AUTHOR = "rstojnic"
EMAIL = "hello@sotabench.com"
URL = "https://sotabench.com"
DESCRIPTION = "Easily benchmark Machine Learning models on selected tasks and datasets - with PyTorch"


setup(
    name=PACKAGE_NAME,
    maintainer=AUTHOR,
    version='0.0.1',
    packages=[PACKAGE_NAME,
              'torchbench.datasets',
              'torchbench.image_classification',
              'torchbench.image_generation',
              'torchbench.object_detection',
              'torchbench.semantic_segmentation'],
    include_package_data=True,
    license=LICENSE,
    description=DESCRIPTION,
    long_description=DESCRIPTION,
    url=URL,
    install_requires=['sotabench', 'torch', 'torchvision'],
)
