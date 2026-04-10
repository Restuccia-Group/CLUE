import torch
import sys
import os
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import numpy as np

from utils import dataset
from models import get_model
from models.nets import  *

from utils.get_activation.hooks import Extractor
from utils.misc import set_deterministic

from hmdepth import datadepth

from methods.odin import ODIN
from methods.energy import Energy

# from  methods.spatial import Spatial
from methods.entropy import Entropy
# from methods.hmd import HMD

import seaborn as sns
import matplotlib.pyplot as plt

from utils.metrics import auroc, fpr_x, aupr

from utils.misc import get_normalization_params
from torchvision.transforms import v2 as transforms

from pytorch_metric_learning import losses

import csv
# import argparse

from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings("ignore")

torch.cuda.empty_cache()
set_deterministic(seed=42, cudnn=False)


class Linear_probe(nn.Module):
    def __init__(self, in_features, out_features = 512, num_classes = 10):
        super(Linear_probe, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(out_features, num_classes)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        
        return x
    
class Linear_fit:
    def __init__(self, model, extractor, num_classes):
        self.model = model
        self.model.eval()
        
        self.extractor = extractor
        self.num_classes = num_classes
        
        #In this case, we are assuming that we will fit one layer at a time
        self.in_features = self.extractor.forward_module_list[0].weight.size()[0]
        
        self.device = next(self.model.parameters())[0].device
        
        self.class_dict = dict()
        for i in range(self.num_classes):
            self.class_dict[i] = i
            
        self.unique_targets = np.unique(np.array(list(self.class_dict.values())))
            
        self.linear_probe = Linear_probe(self.in_features, out_features = 512, num_classes = len(self.unique_targets))
        self.linear_probe.to(self.device)
        self.linear_probe.train()
        
        self.epoch = 50
        self.iniialize_optim_lr_sch()
        
        
    def iniialize_optim_lr_sch(self):
        class_percentage = []
        # u, c = np.unique(np.array(list(self.class_dict.values())), return_counts=True)
        # c = c / np.sum(c)
        # c = 1.0 /c            
        self.loss = nn.CrossEntropyLoss() #losses.SupConLoss(temperature=0.5)
        self.optim = torch.optim.Adam(self.linear_probe.parameters(), lr=0.01, weight_decay=1e-3, amsgrad=True)
        self.lr = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, 10, eta_min=0.0001, last_epoch=-1, verbose=False)
        # self.lr = None
        
    def fit_one_step(self, loader, verbose=False):
        
        for e in range(self.epoch):
            correct = 0
            samples = 0
            total_loss = 0
            for batch_ind, (x,y) in enumerate(loader):
                x, y = x.to(self.device), y.apply_(self.class_dict.get).to(self.device).long()
                
                with torch.no_grad():
                    __ = self.model(x)
                    
                # print(f"Extractor output shape : {self.extractor.forward_activations[0].shape}")
                    
                probe_input = torch.from_numpy(self.extractor.forward_activations[0]).to(self.device)
                
                self.optim.zero_grad()
                with torch.enable_grad():
                    y_pred = self.linear_probe(probe_input)
                    
                                
                current_pred = y_pred.argmax(dim=1, keepdim=True)
                loss_val = self.loss(y_pred, y)
                total_loss = loss_val.item() * len(y)
                loss_val.backward()
                
                self.optim.step()
                if self.lr is not None:
                    self.lr.step()
                
                correct += current_pred.eq(y.view_as(current_pred)).sum().item()
                samples += len(y)
                
                # if verbose:
                    # print(f"Current predictions are : {y_pred}, ground truth are : {y}")
                
                # raise Exception
                
                self.extractor.reset()

            # if verbose:
            # print(f"\nDone with epoch {e}, Loss value is : {total_loss / samples}, \n")
            # print(f"Accuracy at this stage is : {correct / samples}")
            
            
    def get_labels(self, loader):
        
        y_gt = []
        y_pred = []
        
        for batch_ind, (x,y) in enumerate(loader):
            x, y = x.to(self.device), y.apply_(self.class_dict.get).to(self.device).long()
            
            with torch.no_grad():
                y_p = self.model(x)
                probe_input = torch.from_numpy(self.extractor.forward_activations[0]).to(self.device)
                y_p = self.linear_probe(probe_input)
                
            y_gt.extend(list(y.cpu().numpy().ravel()))
            y_pred.extend(list(torch.argmax(y_p, 1).cpu().numpy().ravel()))
            
            self.extractor.reset()
            
        return np.array(y_gt), np.array(y_pred)
            
    def merge_classes(self, confusion_mat, threshold):
        
        cfm = confusion_mat
        error_rate = np.sum(cfm.ravel()[np.argwhere(np.eye(cfm.shape[0]).ravel()==0)]) / np.sum(cfm)
        
        if error_rate>(1.0-threshold) and cfm.shape[0]>2:
            print("\nWill merge now\n")
            label_dict = dict()
            for i in range(cfm.shape[0]):
                label_dict[i] = i
            # max_non_diag_ind = np.argmax(cfm.ravel()[np.argwhere(np.eye(cfm.shape).ravel()==0)])    
            #Merging the classes which are confused the most
            max_error = 0
            for i in range(cfm.shape[0]-1):
                for j in range(i+1, cfm.shape[0]):
                    if cfm[i,j] + cfm[j,i] > max_error:
                        max_error = cfm[i,j] + cfm[j,i]
                        row, col = i, j
          
            #Merge the two classes           
            # if row<col:
            cfm[row , : ] = cfm[row, : ] + cfm[col, : ]
            cfm[ : , row ] = cfm[ : , row ] + cfm[ : , col]
            cfm[col:-1, : ] = cfm[col+1: , : ]
            cfm[ : , col:-1] = cfm[ : , col+1:]
            
            #Merging row and col classes
            label_dict[col] = label_dict[row]
            
            #popping the max class_id from dictionary
            max_class_id = max(list(label_dict.keys()))
            for i in range(col+1, max_class_id+1):
                label_dict[i] -= 1
            
            error_rate = np.sum(cfm.ravel()[np.argwhere(np.eye(cfm.shape[0]).ravel()==0)]) / np.sum(cfm)
            
            print(f"\nLabel dict is now : {label_dict}\n")
            # print(f"\Class dict is now : {self.class_dict}\n")
                
            for key in label_dict:
                for k in self.class_dict:
                    if self.class_dict[k]==key:
                        self.class_dict[k] = label_dict[key]
        print(f"\nClass dict is now : {self.class_dict}\n")


    def fit(self, loader, val_loader, threshold=0.95):
        
        self.fit_one_step(loader, verbose=False)
        
        y_gt, y_pred = self.get_labels(val_loader)
        
        correct = np.sum(y_gt==y_pred)
            
        print(f"The current accuracy is : {correct/len(y_gt)}")
            
        print(f"The accuracy obtained is : {correct / len(y_gt)}")
        return self.linear_probe, len(self.unique_targets)
        
        
        
        
            
        
            
            
            
            
            
            
        
        
        
        
        
        
        
        
        
        
        
        
                
                
                
            
            
            
        
        
    
