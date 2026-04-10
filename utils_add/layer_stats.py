import sys
import os
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import numpy as np

from utils import dataset
from models import get_model

from utils.get_activation.hooks import Extractor
from utils.misc import set_deterministic

from hmdepth import datadepth

from methods.odin import ODIN
from methods.hmd import HMD

from  methods.spatial import Spatial
from methods.entropy import Entropy

from utils.misc import get_normalization_params
from torchvision import transforms

import seaborn as sns
sns.set(palette="husl")
sns.set_style("whitegrid", {'axes.grid' : False})

import matplotlib.pyplot as plt

from utils.metrics import auroc, fpr_x, aupr

import warnings
warnings.filterwarnings("ignore")

def channel_mean_std (model, loader, forward_module_list, extractor, filter_predicted_label=None):
    model.eval()
    device = next(model.parameters())[0].device
    
    forward_activations_mean_dict = dict()
    forward_activations_std_dict = dict()

    for batch_ind, (x,y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        
        with torch.no_grad():
            y_pred = model(x)
            if filter_predicted_label is not None:
                keep = torch.argmax(y_pred,1) == filter_predicted_label
                keep = keep.detach().clone().cpu().numpy()
            else:
                keep = None

        for i in range(len(forward_module_list)):
             
            if len(extractor.forward_activations[i].shape)>2:
                extractor.forward_activations[i] = extractor.forward_activations[i].reshape(extractor.forward_activations[i].shape[0],extractor.forward_activations[i].shape[1], -1)

            if not i in forward_activations_mean_dict.keys():
                if keep is not None:
                    temp_mean = np.mean(extractor.forward_activations[i], -1)[keep, : ]
                    temp_std = np.std(extractor.forward_activations[i], -1)[keep, : ]
                    
                else:
                    temp_mean = np.mean(extractor.forward_activations[i], -1)
                    temp_std = np.std(extractor.forward_activations[i], -1)
                forward_activations_mean_dict[i] = temp_mean
                forward_activations_std_dict[i] = temp_std
            else:
                if keep is not None:
                    temp_mean = np.mean(extractor.forward_activations[i], -1)[keep, : ]
                    temp_std = np.std(extractor.forward_activations[i], -1)[keep, : ]
                    
                else:
                    temp_mean = np.mean(extractor.forward_activations[i], -1)
                    temp_std = np.std(extractor.forward_activations[i], -1)
                forward_activations_mean_dict[i] = np.concatenate((forward_activations_mean_dict[i], temp_mean), 0)
                forward_activations_std_dict[i] = np.concatenate((forward_activations_std_dict[i], temp_std), 0)
        extractor.reset()
        
    return forward_activations_mean_dict, forward_activations_std_dict


def channel_importance (model, loader, module_list, extractor, filter_predicted_label=None):
    model.eval()
    device = next(model.parameters())[0].device
    
    forward_activations_mean_dict = dict()
    forward_activations_std_dict = dict()
    
    loss = torch.nn.CrossEntropyLoss()
    for batch_ind, (x,y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        
        with torch.enable_grad():
            y_pred = model(x)
            if filter_predicted_label is not None:
                keep = torch.argmax(y_pred,1) == filter_predicted_label
                keep = keep.detach().clone().cpu().numpy()
            else:
                keep = None
            loss_val = loss(y_pred, torch.argmax(y_pred, 1))
            loss_val.backward()

        for i in range(len(module_list)):
             
            if len(extractor.forward_activations[i].shape)>3:
                extractor.forward_activations[i] = extractor.forward_activations[i].reshape(extractor.forward_activations[i].shape[0],extractor.forward_activations[i].shape[1], -1)
                
                #Gettign the backward grads
                extractor.backward_grads[i] = extractor.backward_grads[i].reshape(extractor.backward_grads[i].shape[0],extractor.backward_grads[i].shape[1], -1)
                # Grad * Activation
                extractor.forward_activations[i] = np.abs(np.clip(extractor.forward_activations[i], a_min=0, a_max=None) * extractor.backward_grads[i])

            if not i in forward_activations_mean_dict.keys():
                if keep is not None:
                    temp_mean = np.mean(extractor.forward_activations[i], -1)[keep]
                    temp_std = np.std(extractor.forward_activations[i], -1)[keep]
                    
                else:
                    temp_mean = np.mean(extractor.forward_activations[i], -1)
                    temp_std = np.std(extractor.forward_activations[i], -1)
                forward_activations_mean_dict[i] = temp_mean
                forward_activations_std_dict[i] = temp_std
            else:
                if keep is not None:
                    temp_mean = np.mean(extractor.forward_activations[i], -1)[keep]
                    temp_std = np.std(extractor.forward_activations[i], -1)[keep]
                    
                else:
                    temp_mean = np.mean(extractor.forward_activations[i], -1)
                    temp_std = np.std(extractor.forward_activations[i], -1)
                forward_activations_mean_dict[i] = np.concatenate((forward_activations_mean_dict[i], temp_mean), 0)
                forward_activations_std_dict[i] = np.concatenate((forward_activations_std_dict[i], temp_std), 0)
        extractor.reset()
        
    return forward_activations_mean_dict, forward_activations_std_dict
    