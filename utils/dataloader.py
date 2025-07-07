from torch.utils.data import DataLoader, Subset
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_dataset(dataset_name, root="./datasets", train=True, transform=None, download=True, 
                balance=True, alpha=3.0, total_samples=None, seed=None):
    """
    Get a dataset from torchvision.datasets and optionally sample per class using a Dirichlet distribution.

    Args:
    dataset_name (str): Name of the dataset (e.g., 'FashionMNIST', 'CIFAR10', etc.)
    root (str): Root directory of dataset where data will be stored.
    train (bool): If True, creates dataset from training set, otherwise creates from test set.
    transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
    download (bool): If True, downloads the dataset from the internet and puts it in root directory.
    sample_by_dirichlet (bool): If True, samples dataset per class using Dirichlet distribution.
    alpha (float): Concentration parameter for the Dirichlet distribution (higher = more even sampling).
    total_samples (int, optional): Total number of samples to retrieve. If None, uses the original dataset size.
    seed (int, optional): Random seed for reproducibility.

    Returns:
    dataset: A torchvision dataset with modified samples if sample_by_dirichlet is enabled.
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    dataset_class = getattr(datasets, dataset_name)
    dataset = dataset_class(root=root, train=train, transform=transform, download=download)

    # For imbalanced experiment
    if not balance:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Get labels and number of classes
        targets = dataset.targets.clone().detach()
        num_classes = len(torch.unique(targets))

        # Compute original class sizes
        class_counts = torch.bincount(targets)
        total_original = class_counts.sum().item()
        
        # Determine total samples needed
        total_samples = total_samples if total_samples is not None else total_original

        # Sample proportions using Dirichlet distribution
        class_proportions = np.random.dirichlet([alpha] * num_classes)
        
        # Compute target samples per class (rounding ensures whole numbers)
        target_samples_per_class = np.round(class_proportions * total_samples).astype(int)

        sampled_indices = []
        
        for class_idx, num_samples in enumerate(target_samples_per_class):
            class_indices = (targets == class_idx).nonzero(as_tuple=True)[0].tolist()
            if len(class_indices) >= num_samples:
                sampled_indices.extend(np.random.choice(class_indices, num_samples, replace=False))
            else:
                sampled_indices.extend(np.random.choice(class_indices, num_samples, replace=True))

        # Modify dataset in-place
        dataset.data = dataset.data[sampled_indices]
        dataset.targets = torch.tensor([dataset.targets[i] for i in sampled_indices])

    return dataset

def get_data_loader(dataset, labels, batch_size, shuffle=True):
    """
    Get a DataLoader for the specified labels from the dataset.
    
    Args:
    dataset: A torchvision dataset
    labels (list): List of labels to include in the DataLoader
    batch_size (int): How many samples per batch to load
    shuffle (bool): If True, shuffle the data
    
    Returns:
    DataLoader: A DataLoader containing only the specified labels
    """
    if torch.is_tensor(dataset.targets):
        idxs = torch.where(torch.isin(dataset.targets, torch.tensor(labels)))[0]
    else:
        idxs = torch.where(torch.isin(torch.tensor(dataset.targets), torch.tensor(labels)))[0]
    
    subset = Subset(dataset, idxs)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)