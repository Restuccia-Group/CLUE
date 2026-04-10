import sys
import time
import numpy as np
import math
from copy import deepcopy

import torch
import utils
from torch import nn
import torchvision
from torch.nn import functional as F
import os
# from tqdm import tqdm

from .impl import iterative_unlearn

sys.path.append(".")
from imagenet import get_x_y_from_data_dict

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., device='cuda'):
        self.std = std
        self.mean = mean
        self.device = device
        
    def __call__(self, tensor):
        _max = tensor.max()
        _min = tensor.min()
        tensor = tensor + torch.randn(tensor.size()).to(self.device) * self.std + self.mean
        tensor = torch.clamp(tensor, min=_min, max=_max)
        return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def lipschitz_unlearning(lipschitz_model, opt, forget_loader, noise, args, losses, top1, epoch, loader_len):
    start = time.time()
    device = next(lipschitz_model.parameters())[0].device
    for i, sample in enumerate(forget_loader):
        x, target = sample[0].to(device), sample[1].to(device)
        image = x.unsqueeze(0) if x.dim() == 3 else x
        out = lipschitz_model(image)                            
        loss = torch.tensor(0.0, device=device)
        #Build comparison images
        
        for _ in range(100):   
            img2 = noise(deepcopy(x))

            image2 = img2.unsqueeze(0) if img2.dim() == 3 else img2
            
            with torch.no_grad():
                out2 = lipschitz_model(image2)
            # out2 = self.model(image2)
            #ignore batch dimension        
            flatimg, flatimg2 = image.view(image.size()[0], -1), image2.view(image2.size()[0], -1)

            in_norm = torch.linalg.vector_norm(flatimg - flatimg2, dim=1)              
            out_norm = torch.linalg.vector_norm(out - out2, dim=1)
            #K = 0.001 * ((0.4- (out_norm / in_norm)).sum()).abs()#1*((0.08-
            K =  ((out_norm / in_norm).sum()).abs()#pow(2)#  0.1                                                
            loss += K
        loss /= 100
        opt.zero_grad()
        loss.backward()
        opt.step()

        # measure accuracy and record loss
        prec1 = utils.accuracy(out.data, target)[0]
    
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))
    
        if (i + 1) % args.print_freq == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Time {3:.2f}'.format(
                        epoch, i, loader_len, end-start, loss=losses, top1=top1))
            start = time.time()

@iterative_unlearn
def lips_unlearning(data_loaders, model, criterion, optimizer, epoch, args, mask=None, use_mask=True):

    forget_loader = data_loaders["forget"]
    forget_dataset = deepcopy(forget_loader.dataset)

    if args.dataset == "cifar100" or args.dataset == "TinyImagenet":
        # try:
        #     train_dataset.targets = np.random.ones(train_dataset.targets.shape) * -1
        # except:
        #     train_dataset.dataset.targets = np.random.ones(len(train_dataset.dataset.targets)) * -1
    
        train_loader = torch.utils.data.DataLoader(forget_dataset, batch_size=args.batch_size, shuffle=True)
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()

        # switch to train mode
        model.train()

        loader_len = len(forget_loader)
            
        lipschitz_unlearning(model, optimizer, forget_loader, AddGaussianNoise(std=0.8), args, losses, top1, epoch, loader_len)
      
    elif args.dataset == "cifar10" or args.dataset == "svhn":
        train_loader = torch.utils.data.DataLoader(forget_dataset, batch_size=args.batch_size, shuffle=True)
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
      
        # switch to train mode
        model.train()
      
        loader_len = len(forget_loader)

        lipschitz_unlearning(model, optimizer, forget_loader, AddGaussianNoise(std=0.8), args, losses, top1, epoch, loader_len)


    return top1.avg

    
    