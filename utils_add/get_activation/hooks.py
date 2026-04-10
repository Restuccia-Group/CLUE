import os
import sys

import torch
import torch.nn as nn
import numpy as np



class Extractor:
    def __init__(self, forward_module_list, backward_module_list, grad_out=True, grad_in=False, in_numpy=True, 
                 flatten_channels=True, mean_channels=False, discard_neg = True, avg_pool=None):
        self.forward_module_list = forward_module_list
        self.backward_module_list = backward_module_list
        self.in_numpy = in_numpy
        self.flatten_channels = flatten_channels
        self.mean_channels = mean_channels
        self.discard_neg = discard_neg
        
        self.forward_index = list(range(len(forward_module_list))) 
        self.backward_index = list(range(len(forward_module_list)))
        
        self.forward_activation_shape = dict()
        self.backward_activation_shape = dict()
        
        self.forward_activations = dict()
        self.backward_grads = dict()
        
        self.forward_handles = dict()
        self.backward_handles = dict()
        
        self.grad_out = grad_out
        self.grad_in = grad_in
        
        self.avg_pool = avg_pool
        
        
    def back_forward_hook(self, index):
        # print("In Here bak-forward-hook")
        def _hook(module, inputs, output):
            if not self.in_numpy:
                if self.grad_out:
                    self.backward_handles[index] = output.register_hook(self.backward_hook(index))
                if self.grad_in:
                    self.backward_handles[index] = inputs.register_hook(self.backward_hook(index))
            else:
                if self.grad_out:
                    self.backward_handles[index] = output.register_hook(self.numpy_backward_hook(index))
                if self.grad_in:
                    self.backward_handles[index] = inputs.register_hook(self.numpy_backward_hook(index))
        return _hook
    
    def forward_hook(self, index):
        def _hook(module, inputs, output):
            # print(output.requires_grad)
            out = output.detach().clone()
            
            if out.ndim>2 and self.avg_pool is not None:
                size = self.avg_pool if out.size()[2]>self.avg_pool else out.size()[2]
                out = nn.AdaptiveAvgPool2d((size, size))(out)
            
            if self.flatten_channels:
                out = out.flatten(1)
            elif out.ndim>=3:
                out = out.flatten(2)
            if self.discard_neg:
                out = torch.clamp(out, min=0)
            if self.mean_channels:
                if out.ndim>2:
                    out = out.mean(2)
            self.forward_activations[index] = out
            if not index in self.forward_activation_shape.keys():
                self.forward_activation_shape[index] = out.size()
                
        return _hook
    
    def backward_hook(self, index):
        def _hook(grad_output):
            out = grad_output.detach().clone()
            if self.flatten_channels:
                out = out.flatten(1)
            self.backward_grads[index] = out
            if not index in self.backward_activation_shape.keys():
                self.backward_activation_shape[index] = out.size()
                
        return _hook
            
    def numpy_forward_hook(self, index):
        def _hook(module, inputs, output):
            out = output.detach().clone()
            
            if out.ndim>2 and self.avg_pool is not None:
                size = self.avg_pool if out.size()[2]>self.avg_pool else out.size()[2]
                out = nn.AdaptiveAvgPool2d((size, size))(out)
                
            if self.flatten_channels:
                out = out.flatten(1)
            elif out.ndim>=3:
                out = out.flatten(2)
            if self.discard_neg:
                out = torch.clamp(out, min=0)            
            if self.mean_channels:
                if out.ndim>2:
                    out = out.mean(2)
            self.forward_activations[index] = out.cpu().numpy()
            if not index in self.forward_activation_shape.keys():
                self.forward_activation_shape[index] = out.cpu().numpy().shape        
        return _hook
    
    def numpy_backward_hook(self, index):
        def _hook(grad_output):
            out = grad_output.detach().clone()
            if self.flatten_channels:
                out = out.flatten(1)
            self.backward_grads[index] = out.cpu().numpy()
            # print("Logged backward grads")
            if not index in self.backward_activation_shape.keys():
                self.backward_activation_shape[index] = out.cpu().numpy().shape 
                
        return _hook
                
                
    def register_forward_hooks(self):
        for ind, module in enumerate(self.forward_module_list):
            if self.in_numpy:
                h = module.register_forward_hook(self.numpy_forward_hook(ind))
                self.forward_handles[ind] = h
            else:
                h = module.register_forward_hook(self.forward_hook(ind))
                self.forward_handles[ind] = h
                
    def register_backward_hooks(self):
        for ind, module in enumerate(self.backward_module_list):
            h = module.register_forward_hook(self.back_forward_hook(ind))
            self.forward_handles[ind] = h
            
    def clear_forward_hooks(self):
        for h_ind in self.forward_handles:
            self.forward_handles[h_ind].remove()
            
    def clear_backward_hooks(self):
        for h_ind in self.backward_handles:
            self.backward_hooks[h_ind].remove()
            
    def clear_all_hooks(self):
        self.clear_forward_hooks()
        self.clear_backward_hooks()
            
    def reset_all(self):
        self.clear_forward_hooks()
        self.clear_backward_hooks()
        
        # self.forward_index = list(range(len(self.forward_module_list))) 
        # self.backward_index = list(range(self.backward_module_list))
        
        self.forward_activation_shape = dict()
        self.backward_activation_shape = dict()
        
        self.forward_activations = dict()
        self.backward_grads = dict()
        
        self.forward_handles = dict()
        self.backward_handles = dict()
        
    def reset(self):
        # self.clear_forward_hooks()
        # self.clear_backward_hooks()
        
        # self.forward_index = list(range(len(self.forward_module_list))) 
        # self.backward_index = list(range(self.backward_module_list))
        
        self.forward_activation_shape = dict()
        self.backward_activation_shape = dict()
        
        self.forward_activations = dict()
        self.backward_grads = dict()
        
        self.forward_handles = dict()
        self.backward_handles = dict()
        
    #Preprocessor function in case of need
    def preprocessor(self, x, y, model, loss=None):
        
        if loss is None:
            loss = nn.CrossEntropyLoss()
            
        x.requires_grad = True
        with torch.enable_grad():
            y_pred = model(x)
            
        loss_val = loss(y_pred,torch.argmax(y_pred,1))
        loss_val.backward()
        
        x = x - (0/256) * x.grad.detach().clone().sign()
        
        
        return x
    
    def get_activations(self, model, loader, device, preprocessor=False, post_process=None, from_class = None):
        forward_activations_ = dict()
        loss = torch.nn.CrossEntropyLoss()
        for batch_ind, (x,y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            
            if preprocessor:
                x = self.preprocessor(x, y)
                        
            with torch.enable_grad():
                y_pred = model(x)
                loss_val = loss(y_pred, torch.argmax(y_pred, 1))
                loss_val.backward()
                
            if from_class is not None:
                indices = torch.eq(torch.argmax(y_pred, 1), from_class).cpu().numpy() #We are filtering anything that is out of from_class
                
            for i in self.forward_activations:
                if not i in forward_activations_.keys():
                    forward_activations_[i] = list()
                
                if from_class is not None:
                    filter_importance = np.abs(self.forward_activations[i] * self.backward_grads[i])[indices, : ]   #This just gets the filter importance
                    activations = self.forward_activations[i][indices, : ]
                else:
                    filter_importance = np.abs(self.forward_activations[i] * self.extractor.backward_grads[i])
                    activations = self.forward_activations[i]
                
                if len(filter_importance.shape)>2:
                    filter_importance = np.mean(filter_importance, -1)
                    activations = np.mean(activations, -1)

                forward_activations_[i].append(np.concatenate((filter_importance, activations), -1))
                # print(type(activations[0,0]))
                # forward_activations_[i].append(activations)
            self.reset()
                
                
        final_train = None
        for ind in forward_activations_:
            f_train = None
            for act in forward_activations_[ind]:
                if f_train is None:
                    f_train = np.array(act)
                else:
                    f_train = np.concatenate((f_train, np.array(act)), axis=0)
            if final_train is None:
                final_train = f_train
            else:
                final_train = np.concatenate((final_train, f_train), -1)

        return final_train 
    
    def get_activations_(self, model, loader, device, preprocessor=False, post_process=None):
        model.eval()
        forward_activations_dict = dict()

        for batch_ind, (x,y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            
            if preprocessor:
                x = self.preprocessor(x, y, model)
            
            with torch.no_grad():
                y_pred = model(x)
            for i in self.forward_activations:
                if not i in forward_activations_dict.keys():
                    forward_activations_dict[i] = list()
                forward_activations_dict[i].append(self.forward_activations[i])
            self.reset()
                
        return_dict = dict()
        
        for ind in forward_activations_dict:
            f_train = None
            for act in forward_activations_dict[ind]:
                if len(act.shape)>2:
                    act = np.reshape(act, (act.shape[0], -1))
                if f_train is None:
                    if post_process is not None:
                        if post_process=="mean":
                            act = np.mean(act, 1)
                        elif post_process=="median":
                            act = np.median(act, 1)
                        elif post_process=="max":
                            act = np.max(act, 1)
                        elif post_process=="std":
                            act = np.std(act, 1)
                        
                    f_train = np.array(act)
                else:
                    if post_process is not None:
                        if post_process=="mean":
                            act = np.mean(act, 1)
                        elif post_process=="median":
                            act = np.median(act, 1)
                        elif post_process=="max":
                            act = np.max(act, 1)
                        elif post_process=="std":
                            act = np.std(act, 1)
                            
                    f_train = np.concatenate((f_train, np.array(act)), axis=0)

        return f_train 
        
        
        
        
        
        
            