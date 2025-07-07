#!/usr/bin/env python
# coding: utf-8

import argparse
from utils.dpi_class import DPI_CLASS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPI-RG Experiment Runner")
    parser.add_argument('--dataset', type=str, required=True, choices=['FashionMNIST', 'CIFAR10'],
                        help='Dataset to use')
    args = parser.parse_args()

    model = DPI_CLASS(
        dataset_name=args.dataset,
        z_dim=5,
        lr_I=4e-4,
        lr_G=4e-4,
        lr_f=4e-4,
        weight_decay=0.01,
        batch_size=500,
        epochs1=50,
        epochs2=100,
        lambda_mmd=2.0,
        lambda_gp=0.1,
        lambda_power=0.6,
        eta=2.5,
        std=0.5,
        present_label=[0,1,2,3,4,5,6,7,8,9],
        critic_iter=8,
        critic_iter_f=8,
        critic_iter_p=8,
        decay_epochs=40,
        gamma=0.2,
        balance=True,
    )

    model.train()
    model.validate_w_classifier()
