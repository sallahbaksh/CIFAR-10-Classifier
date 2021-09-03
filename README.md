# CIFAR-10-Classifier

## Introduction

Four deep learning models (ANN, MLP, CNN, and GAN) were implemented to classify images from the [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html) dataset.

## Getting Started

### Prerequisites

The following dependencies are required:<br/>

* numpy

* os
* pickle
* concurrent.futures
* tqdm
    * trange
* skimage
    * transform
* sklearn
    * accuracy_score
* softmax
    * Softmax
* matplotlib
    * pyplot

If using Anaconda with Python 3.8+, everything above is included except `tqdm` and `concurrent.futures`. 

To add `tqdm`, run:

```bash
conda install -c conda-forge tqdm
```

To add `concurrent.futures`, run:
```bash
conda install -c anaconda futures
```

Else, individually install these libraries using `pip install`.

### Running The Python Scripts and Notebooks

Given the `.idea` folder, the project can be set up in PyCharm Professional. However, using other IDEs such as Jupyter
Notebook or JupyterLab will also suffice.
