# DPI-RG

This repository contains the experiment code used for the project Distribution-free Predictive Inference using Roundtrip Generative Models (DPI-RG) published on [arXiv](https://arxiv.org/abs/2205.10732).

## Usage
You can recreate the environment we used for the experiments by running `conda env create -f environment.yml`.

To conduct an experiment for Fashion-MNIST, please run `nohup python -u fmnist_main.py > fmnist.log &`.

Similarly for CIFAR10, please run `nohup python -u cifar10_main.py > cifar10.log &`.

## Results
The coverage and average set size will be printed out as the validation process completes.
A folder named `graphs/timestamp` will show up as you validate the model with testing dataset. You can view the histograms of p-values and test statistics in it.  

## Acknowledgement
We would like to acknowledge the authors of [inferential Wasserstein GAN](https://academic.oup.com/jrsssb/article/84/1/83/7056079), as part of our code is adapted from their code repository.
