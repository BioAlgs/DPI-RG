import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

import os
import json
import torch
import torch.nn as nn
from .dataloader import get_dataset
from .mnist import I_MNIST, G_MNIST, f_MNIST
from .cifar10 import I_CIFAR10, G_CIFAR10, f_CIFAR10

class BaseDPI:
    def __init__(self, dataset_name, lr_G, lr_I, lr_f, weight_decay, batch_size, 
                 lambda_mmd, lambda_gp, eta, std, present_label, missing_label=[], 
                 critic_iter=3, critic_iter_f=3, decay_epochs=None, 
                 gamma=0.2, balance=True, device=None, timestamp=None):
        # Common initialization code
        self.dataset_name = dataset_name
        self.train_gen = get_dataset(dataset_name, balance=balance, train=True)
        self.test_gen = get_dataset(dataset_name, balance=balance, train=False)
        self.lr_I = lr_I
        self.lr_G = lr_G
        self.lr_f = lr_f
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.lambda_mmd = lambda_mmd
        self.lambda_gp = lambda_gp
        self.eta = eta
        self.std = std
        self.critic_iter = critic_iter
        self.critic_iter_f = critic_iter_f
        self.decay_epochs = decay_epochs
        self.gamma = gamma
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.present_label = present_label
        self.missing_label = missing_label
        self.all_label = self.present_label + self.missing_label
        self.z_dim = len(present_label) if not hasattr(self, 'z_dim') else self.z_dim
        self.models = {}
        self.optimizers = {}
        self.T_train = None
        
        # Set timestamp and mode
        if timestamp is None:
            self.timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
            self.validation_only = False
        else:
            self.timestamp = timestamp
            self.validation_only = True
        
        print(f"Current timestamp: {self.timestamp}")

        # Save folder paths as instance variables
        self.graphs_folder = f'graphs/{self.timestamp}'
        self.params_folder = f'params/{self.timestamp}'

        # Create the folders for saving plots
        os.makedirs(self.graphs_folder, exist_ok=True)
        os.makedirs(self.params_folder, exist_ok=True)

        # Determine model suffix once based on dataset_name
        if self.dataset_name in ["MNIST", "FashionMNIST"]:
            self.model_suffix = "MNIST"
            self.nc = 1
            self.img_size = 28
        elif self.dataset_name == "CIFAR10":
            self.model_suffix = "CIFAR10"
            self.nc = 3
            self.img_size = 32
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
        # Save the parameters
        if timestamp is None:
            self.save_parameters()
        
        # Retrieve model classes once and store them as instance attributes
        self.netI_class = globals().get(f"I_{self.model_suffix}")
        self.netG_class = globals().get(f"G_{self.model_suffix}")
        self.netf_class = globals().get(f"f_{self.model_suffix}")

        # Ensure all required models are found
        if not all([self.netI_class, self.netG_class, self.netf_class]):
            raise ValueError(f"Model classes I_{self.model_suffix}, G_{self.model_suffix}, f_{self.model_suffix} not found.")

        # Initialize or load models
        self.setup_models()



    def setup_models(self):
        # This method should be implemented in child classes
        raise NotImplementedError("Subclass must implement abstract method")

    def train(self):
        # Common training logic
        raise NotImplementedError("Subclass must implement abstract method")

    def validate(self):
        # Common validation logic
        raise NotImplementedError("Subclass must implement abstract method")

    def save_model(self):
        # Common model saving logic
        raise NotImplementedError("Subclass must implement abstract method")

    def load_inverse_model(self):
        # Common model loading logic
        raise NotImplementedError("Subclass must implement abstract method")

    def get_fake_zs(self, train_loader):
        # Common logic for getting fake zs
        raise NotImplementedError("Subclass must implement abstract method")
    
    def save_loss_plots(self, GI_losses, MMD_losses, f_losses, GP_losses):
        # Common logic for saving loss plots
        raise NotImplementedError("Subclass must implement abstract method")

    def generate_fixed_noise(self):
        # Common logic for saving loss plots
        raise NotImplementedError("Subclass must implement abstract method")

    def display_fake_images(self, netG):
        # Common logic for saving loss plots
        raise NotImplementedError("Subclass must implement abstract method")
    
    def save_parameters(self):
        """
        Save all parameters of the instance to a file, skipping non-serializable parameters.
        """
        params = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                try:
                    # Try to serialize the value
                    json.dumps(value)
                    params[key] = value
                except (TypeError, ValueError):
                    # Skip non-serializable parameters silently
                    continue

        # Sort parameters by name
        sorted_params = dict(sorted(params.items()))

        # Save the selected parameters to a JSON file
        with open(f'{self.params_folder}/hyper_param.json', 'w') as file:
            json.dump(sorted_params, file, indent=4)
    
    @staticmethod
    def next_batch(data_iter, train_loader):
        try:
            return next(data_iter)
        except StopIteration:
            return next(iter(train_loader))
        
    @staticmethod
    def freeze_batch_norm_layers(net):
        for module in net.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                module.eval()  # Set the batch norm layer to evaluation mode
                for param in module.parameters():
                    param.requires_grad = False  # Disable gradient updates for batch norm parameters


    def visualize_p(self, all_p_vals, classes, path):
        # print('-'*100, '\n', ' ' * 45, 'p-values', '\n', '-'*100, sep = '')
        present_label = self.present_label
        all_label = self.all_label

        if len(present_label) == 1:
            fig, axs = plt.subplots(len(present_label), len(all_label), 
                                    figsize=(5*len(all_label), 5*len(present_label)))

            matplotlib.rc('xtick', labelsize=15) 
            matplotlib.rc('ytick', labelsize=15) 

            for i, lab in enumerate(all_label):
                p_vals_class = all_p_vals[lab]
                axs[i].set_xlim([0, 1])
                _ = axs[i].hist(p_vals_class[present_label[0]])
                prop = np.sum(np.array(p_vals_class[present_label[0]] <= 0.05) / len(p_vals_class[present_label[0]]))
                prop = np.round(prop, 4)
                if all_label[i] == present_label[0]:
                    axs[i].set_title('Type I Error: {}'.format(prop), fontsize = 20)
                else:
                    axs[i].set_title('Power: {}'.format(prop), fontsize = 20)
                if i == 0:
                    axs[i].set_ylabel(classes[present_label[0]], fontsize = 25)
                axs[i].set_xlabel(classes[all_label[i]], fontsize = 25)
        else:
            fig, axs = plt.subplots(len(present_label), len(all_label), 
                                    figsize=(5*len(all_label), 5*len(present_label)))

            matplotlib.rc('xtick', labelsize=15) 
            matplotlib.rc('ytick', labelsize=15) 

            for i, val_lab in enumerate(all_label):
                p_vals_class = all_p_vals[val_lab]
                for j, train_lab in enumerate(present_label):
                    axs[j, i].set_xlim([0, 1])
                    _ = axs[j, i].hist(p_vals_class[train_lab])
                    prop = np.sum(np.array(p_vals_class[train_lab] <= 0.05) / len(p_vals_class[train_lab]))
                    prop = np.round(prop, 4)
                    if all_label[i] == present_label[j]:
                        axs[j, i].set_title('Type I Error: {}'.format(prop), fontsize = 20)
                    else:
                        axs[j, i].set_title('Power: {}'.format(prop), fontsize = 20)
                    if i == 0:
                        axs[j, i].set_ylabel(classes[present_label[j]], fontsize = 25)
                    if j == len(present_label) - 1:
                        axs[j, i].set_xlabel(classes[all_label[i]], fontsize = 25)
        
        fig.supylabel('Training', fontsize = 25)
        fig.supxlabel('Validation', fontsize = 25)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        

    def visualize_T(self, all_fake_Cs, classes, path):
        # print('-'*100, '\n', ' ' * 45, 'fake numbers', '\n', '-'*100, sep = '')
        present_label = self.present_label
        all_label = self.all_label
        
        if len(present_label) == 1:
            fig, axs = plt.subplots(1, len(all_label), 
                                    figsize=(5*len(all_label), 5*len(present_label)))

            matplotlib.rc('xtick', labelsize=15) 
            matplotlib.rc('ytick', labelsize=15) 
            llim = np.min([np.min(vals[present_label[0]]) for vals in all_fake_Cs.values()])
            rlim = np.quantile(np.concatenate([vals[present_label[0]] for vals in all_fake_Cs.values()]), 0.98)
            for i, lab in enumerate(all_label):
                fake_Cs = all_fake_Cs[lab]
                axs[i].set_ylabel(classes[lab], fontsize = 25)
                axs[i].set_xlim([llim, rlim])
                _ = axs[i].hist(fake_Cs[present_label[0]])
                # axs[i].set_title('Label {} from Label {}\'s net'.format(lab, present_label[0]), fontsize = 20)

                if i == len(all_label) - 1:
                    axs[i].set_xlabel(classes[present_label[0]], fontsize = 25)
        else:
            fig, axs = plt.subplots(len(present_label), len(all_label), 
                                    figsize=(5*len(all_label), 5*len(present_label)))

            matplotlib.rc('xtick', labelsize=15) 
            matplotlib.rc('ytick', labelsize=15) 
            llim = np.min([np.min(list(vals.values())) for vals in all_fake_Cs.values()])
            rlim = np.quantile(np.concatenate([np.concatenate(list(vals.values())) for vals in all_fake_Cs.values()]), 0.95)
            for i, val_lab in enumerate(all_label):
                fake_Cs = all_fake_Cs[val_lab]
                for j, train_lab in enumerate(present_label):
                    axs[j, i].set_xlim([llim, rlim])
                    _ = axs[j, i].hist(fake_Cs[train_lab])
                    # axs[j, i].set_title('Label {} from Label {}\'s net'.format(val_lab, train_lab))
                    if i == 0:
                        axs[j, i].set_ylabel(classes[train_lab], fontsize = 25)
                    if j == len(present_label) - 1:
                        axs[j, i].set_xlabel(classes[val_lab], fontsize = 25)
        
        fig.supylabel('Training', fontsize = 25)
        fig.supxlabel('Validation', fontsize = 25)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

