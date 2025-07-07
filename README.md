# DPI-RG

This repository contains the experiment code for the project **Distribution-free Predictive Inference using Roundtrip Generative Models (DPI-RG)**, as published on [arXiv](https://arxiv.org/abs/2205.10732).

## Installation
To set up the environment required for the experiments, run:
```{bash}
conda env create -f environment.yml
conda activate dpi
```

## File Structure
- `main.py`: Main script for experiments.
- `utils/`: Contains core classes, model definitions, and utility functions.
- `graphs/`: Output directory for plots and visualizations.
- `params/`: Stores hyperparameter settings and network parameters for each experiment run.
- `environment.yml`: Conda environment specification.

## Usage
### Running Experiments
- Fashion-MNIST
```{python}
python main.py --dataset FashionMNIST > fmnist.log 
```
- CIFAR10
```{python}
python main.py --dataset CIFAR10 > cifar10.log 
```

### Custom Experiments
You can modify the main script (main.py) to adjust hyperparameters or experiment settings. See the script comments for parameter descriptions.

- Imbalanced Data:
Set the `balance` argument to `False` when creating the `DPI_CLASS` instance.

- Missing Classes:
Change the `present_label` argument to include only the classes you want to use for training.
For example, to exclude class 9:
```{python}
model = DPI_CLASS(
    ...,
    present_label=[0,1,2,3,4,5,6,7,8],  # Exclude class 9
    balance=False,                      # Use imbalanced data
    ...
)
```

## Results

- Coverage and Set Size
During validation, the script prints the coverage and average set size to the console/log.

- Visualization
After validation, a folder named `graphs/<timestamp>` will be created.
This folder contains histograms of p-values and test statistics for further analysis.

## Reproducibility
The experiments were conducted on a server with four NVIDIA RTX A4000 GPUs.

For Reproducibility: 
- Each run saves parameters and random seeds in `params/<timestamp>/hyper_param.json`.
- To replicate a specific run, load these saved parameters and set the corresponding `<timestamp>` in the script.

## Acknowledgements
We would like to acknowledge the authors of [inferential Wasserstein GAN](https://academic.oup.com/jrsssb/article/84/1/83/7056079), as part of our code is adapted from their code repository.

--- 
For questions or issues, please open an issue on GitHub or contact the authors.
