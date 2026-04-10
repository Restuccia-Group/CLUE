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

# defining the noise structure
class Noise(nn.Module):
    def __init__(self, *dim):
        super().__init__()
        self.noise = torch.nn.Parameter(torch.randn(*dim), requires_grad = True)
        
    def forward(self):
        return self.noise

# This trains the noise to maximize the  output probability of the forget class(es).
def train_noise(model, args):
    noises = {}
    forget_class = args.class_to_replace if isinstance(args.class_to_replace, list) else [args.class_to_replace]
    for cls in forget_class:
        print("Optiming loss for class {}".format(cls))
        noises[cls] = Noise(args.batch_size, 3, 32, 32).cuda()
        opt = torch.optim.Adam(noises[cls].parameters(), lr = 0.1)

        num_epochs = 5
        num_steps = 8
        class_label = cls
        for epoch in range(args.num_noise_epochs):
            total_loss = []
            for batch in range(num_steps):
                inputs = noises[cls]()
                labels = torch.zeros(args.batch_size).cuda()+class_label
                outputs = model(inputs)
                loss = -F.cross_entropy(outputs, labels.long()) + 0.1*torch.mean(torch.sum(torch.square(inputs), [1, 2, 3]))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss.append(loss.cpu().detach().numpy())
            print("Loss: {}".format(np.mean(total_loss)))

    return noises

#This uses the generated nosie to impair the model to forget the classes
def impair(model, noises, opt, args, losses, top1, epochs, start):
    batch_size = 256
    noisy_data = []
    num_batches = 20
    class_num = 0

    forget_class = args.class_to_replace if isinstance(args.class_to_replace, list) else [args.class_to_replace]

    for cls in forget_class:
        for i in range(num_batches):
            batch = noises[cls]().cpu().detach()
            for i in range(batch[0].size(0)):
                noisy_data.append((batch[i], torch.tensor(class_num)))

    #We are only using the noise and not the retain set to adapt it in our setting
    noisy_loader = torch.utils.data.DataLoader(noisy_data, batch_size=args.batch_size, shuffle = True)

    # opt = torch.optim.Adam(model.parameters(), lr = 0.02)

    for epoch in range(1):  
        model.train(True)
        running_loss = 0.0
        running_acc = 0
        for i, data in enumerate(noisy_loader):
            inputs, labels = data
            inputs, labels = inputs.cuda(),labels.cuda()

            opt.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            opt.step()

            # measure accuracy and record loss
            prec1 = utils.accuracy(outputs.data, labels)[0]
        
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
        
            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Time {3:.2f}'.format(
                            epoch, i, len(noisy_loader), end-start, loss=losses, top1=top1))
                start = time.time()

@iterative_unlearn
def unsir(data_loaders, model, criterion, optimizer, epoch, args, mask=None, use_mask=True):

    if args.dataset == "cifar100" or args.dataset == "TinyImagenet":

        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()

        # switch to train mode
        model.train()

        start = time.time()
        noise = train_noise(model, args)
            
        impair(model, noise, optimizer, args, losses, top1, epoch, start)
      
    elif args.dataset == "cifar10" or args.dataset == "svhn":
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()

        # switch to train mode
        model.train()

        start = time.time()
        noise = train_noise(model, args)
            
        impair(model, noise, optimizer, args, losses, top1, epoch, start)


    return top1.avg
    