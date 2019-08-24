import io
from setuptools import setup
from torchbench.version import __version__

name = "torchbench"
author = "Atlas ML"
author_email = "hello@sotabench.com"
license = "Apache-2.0"
url = "https://sotabench.com"
description = (
    "Easily benchmark Machine Learning models on selected tasks and datasets"
    " - with PyTorch"
)


def get_requirements():
    with io.open("requirements.txt") as f:
        return [
            line.strip()
            for line in f.readlines()
            if not line.strip().startswith("#")
        ]


setup(
    name=name,
    version=__version__,
    author=author,
    author_email=author_email,
    maintainer=author,
    maintainer_email=author_email,
    description=description,
    long_description=io.open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url=url,
    platforms=["Windows", "POSIX", "MacOSX"],
    license=license,
    packages=[
        name,
        "torchbench.datasets",
        "torchbench.image_classification",
        "torchbench.image_generation",
        "torchbench.language_modelling",
        "torchbench.object_detection",
        "torchbench.semantic_segmentation",
    ],
    include_package_data=True,
    install_requires=get_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)

