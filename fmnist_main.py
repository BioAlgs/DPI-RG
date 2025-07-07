#!/usr/bin/env python
# coding: utf-8

from utils.dpi_class import DPI_CLASS

if __name__ == "__main__":
    # Model and dataset configuration
    model = DPI_CLASS(
        dataset_name='FashionMNIST',  # Dataset to use
        z_dim=5,                      # Latent dimension

        # Learning rates
        lr_I=4e-4,                    # Learning rate for I
        lr_G=4e-4,                    # Learning rate for G
        lr_f=4e-4,                    # Learning rate for f

        weight_decay=0.01,            # Weight decay for optimizer
        batch_size=500,               # Batch size

        # Training epochs
        epochs1=50,                   # Number of epochs for phase 1
        epochs2=100,                  # Number of epochs for phase 2

        # Loss and regularization parameters
        lambda_mmd=2.0,               # MMD loss weight
        lambda_gp=0.1,                # Gradient penalty weight
        lambda_power=0.6,             # Power loss weight
        eta=2.5,                      # Eta parameter
        std=0.5,                      # Standard deviation for noise

        present_label=[0,1,2,3,4,5,6,7,8,9],  # Labels present in training

        # Critic iterations
        critic_iter=8,
        critic_iter_f=8,
        critic_iter_p=8,

        # Learning rate decay
        decay_epochs=40,              # Epochs before decay
        gamma=0.2,                    # Decay factor

        balance=True,                 # Balanced case or not
    )

    model.train()
    model.validate_w_classifier()

