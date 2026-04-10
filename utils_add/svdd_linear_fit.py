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

import seaborn as sns
import matplotlib.pyplot as plt

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
        self.linear1 = nn.Linear(in_features, in_features//2)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features//2, out_features)
        self.relu = nn.ReLU()
        self.linear3 = nn.Linear(out_features, num_classes)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x_ = self.linear2(x)
        x = self.relu(x_)
        x = self.linear3(x)
        
        return x, x_
    
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, feature_vectors, labels):
        # Normalize feature vectors
        feature_vectors_normalized = F.normalize(feature_vectors, p=2, dim=1)
        # Compute logits
        logits = torch.div(
            torch.matmul(
                feature_vectors_normalized, torch.transpose(feature_vectors_normalized, 0, 1)
            ),
            self.temperature,
        )
        return losses.NTXentLoss(temperature=0.07)(logits, torch.squeeze(labels))
    
class Linear_fit:
    def __init__(self, model, extractor, num_classes):
        self.model = model
        self.model.eval()
        
        self.extractor = extractor
        self.num_classes = num_classes
        
        #In this case, we are assuming that we will fit one layer at a time
        self.in_features = sum([module.weight.size()[0] for module in self.extractor.forward_module_list])
        
        self.device = next(self.model.parameters())[0].device
        
        self.class_dict = dict()
        for i in range(self.num_classes):
            self.class_dict[i] = i
            
        self.unique_targets = np.unique(np.array(list(self.class_dict.values())))
            
        self.linear_probe = Linear_probe(self.in_features, out_features = 512, num_classes = len(self.unique_targets))
        self.linear_probe.to(self.device)
        self.linear_probe.train()
        
        self.epoch = 40
        self.iniialize_optim_lr_sch()
        
        self.cross_loss = nn.CrossEntropyLoss()
        self.kl_div_loss = nn.KLDivLoss(reduction="batchmean")
        
        self.center = None
        self.c_momentum = 0.8
        self.stop_c_update = 10
        
        self.supcon = SupervisedContrastiveLoss()
        
        
    def loss(self, y_pred, y, y_pred_base, x_int, center, lambda1=0.1, lambda2=10):
        #(1 - lambda1 - lambda2) * 
        loss_val = self.supcon(x_int, y) + lambda1 * self.kl_div_loss(nn.functional.log_softmax(y_pred, dim=-1), nn.functional.softmax(y_pred_base, dim=-1) ) #+ 0.5 * lambda2 * torch.mean(torch.pow(torch.norm(x_int - self.center, 2,  -1), 2))
        
        return loss_val
        
    def iniialize_optim_lr_sch(self):
        
        class_percentage = []         
        # self.loss = nn.CrossEntropyLoss() #losses.SupConLoss(temperature=0.5)
        self.optim = torch.optim.Adam(self.linear_probe.parameters(), lr=0.01, weight_decay=1e-3, amsgrad=True)
        self.lr = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, 10, eta_min=0.0001, last_epoch=-1, verbose=False)
        
    def get_probe_input(self):
        f = None
        for i in self.extractor.forward_activations:
            if f is None:
                f = self.extractor.forward_activations[i]
            else:
                f = np.concatenate((f, self.extractor.forward_activations[i]), -1)
                
        return f
        
    def fit_one_step(self, loader, val_loader=None, verbose=False):
        
        for e in range(self.epoch):
            correct = 0
            samples = 0
            total_loss = 0
            
            val_correct = 0
            val_samples = 0
            
            approx_average_radius = 0
            for batch_ind, (x,y) in enumerate(loader):
                x, y = x.to(self.device), y.apply_(self.class_dict.get).to(self.device).long()
                
                with torch.no_grad():
                    y_pred_base = self.model(x)
                    
                # print(f"Extractor output shape : {self.extractor.forward_activations[0].shape}")
                    
                probe_input = torch.from_numpy(self.get_probe_input()).to(self.device)
                
                self.optim.zero_grad()
                with torch.enable_grad():
                    y_pred, x_int = self.linear_probe(probe_input)
                    
                if e<self.stop_c_update:
                    self.center = torch.mean(x_int, 0) if self.center is None else (1-self.c_momentum) * self.center + self.c_momentum * torch.mean(x_int, 0) 
                    
                current_pred = y_pred.argmax(dim=1, keepdim=True)
                loss_val = self.loss(y_pred, y, y_pred_base, x_int, self.center)
                total_loss += loss_val.item() * len(y)
                loss_val.backward(retain_graph=True)
                
                self.optim.step()
                if self.lr is not None:
                    self.lr.step()
                
                val_correct += current_pred.eq(y.view_as(current_pred)).sum().item()
                val_samples += len(y)
                
                self.extractor.reset()
                
            
            if val_loader is not None:
                for batch_ind, (x,y) in enumerate(val_loader):
                    x, y = x.to(self.device), y.to(self.device)
                    
                    with torch.no_grad():
                        y_pred_base = self.model(x)
                        
                    probe_input = torch.from_numpy(self.get_probe_input()).to(self.device)
                    
                    y_pred, x_int = self.linear_probe(probe_input)
                    
                    current_pred = y_pred.argmax(dim=1, keepdims=True)
                    correct += current_pred.eq(y.view_as(current_pred)).sum().item()
                    samples += len(y)
                    approx_average_radius += torch.mean(torch.pow(torch.norm(x_int - self.center, 2, -1), 2)).item()
            
                    self.extractor.reset()
            print(f"The current accuracy at epoch {e} : {correct/samples} and average radius is : {approx_average_radius/batch_ind}")
            
    def get_labels(self, loader):
        
        y_gt = []
        y_pred = []
        
        for batch_ind, (x,y) in enumerate(loader):
            x, y = x.to(self.device), y.apply_(self.class_dict.get).to(self.device).long()
            
            with torch.no_grad():
                y_p = self.model(x)
                
                probe_input = torch.from_numpy(self.get_probe_input()).to(self.device)
                y_p, x_int = self.linear_probe(probe_input)
                
            y_gt.extend(list(y.cpu().numpy().ravel()))
            y_pred.extend(list(torch.argmax(y_p, 1).cpu().numpy().ravel()))
            
            self.extractor.reset()
            
        return np.array(y_gt), np.array(y_pred)
            
    def fit(self, loader, val_loader, threshold=0.95):
        
        self.fit_one_step(loader, val_loader, verbose=False)
        
        # y_gt, y_pred = self.get_labels(val_loader)
        
        # correct = np.sum(y_gt==y_pred)
            
        
            
        # print(f"The accuracy obtained is : {correct / len(y_gt)}")
        return self.linear_probe, self.center
        
        
        
        
            
        
            
            
            
            
            
            
        
        
        
        
        
        
        
        
        
        
        
        
                
                
                
            
            
            
        
        
    
