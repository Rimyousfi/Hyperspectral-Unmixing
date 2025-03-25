# Hyperspectral-Unmixing
MiSciCNet and CyCuNet Deep Learning models to unmix hyperspectral images


Overview

This project contains Jupyter notebooks for training and evaluating two models:

CycuNet: A deep learning model for hyperspectral unmixing.
MiSiCNet: A model that uses single-image training for endmember extraction and abundance estimation.
Additionally, a Python script (MiSiCNet_tools.py) provides preprocessing functions for MiSiCNet.


Installation

To run the notebooks, install the required libraries:
pip install numpy scipy torch matplotlib scikit-learn torchvision

Python & Library Versions

Python: 3.12.3
NumPy: 1.26.4
Torch: 2.6.0+cpu
Matplotlib: 3.9.2
SciPy: 1.13.1


This project is based on the following two articles: Behnood Rasti et al. “Misicnet: Minimum simplex convolutional network for deep
hyperspectral unmixing” and Lianru Gao et al. “CyCU-Net: Cycle-consistency unmixing network by learning cascaded autoencoders”.
