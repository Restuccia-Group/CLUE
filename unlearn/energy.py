import sys
import time
import numpy as np
from copy import deepcopy

import torch
import utils

from .impl import iterative_unlearn

sys.path.append(".")
from imagenet import get_x_y_from_data_dict


def create_mask_from_gradients(model, test_loader, threshold=0.15):
    """
    Creates a mask for the weights of a model based on gradient information from the test data.
    Args:
        model: The neural network model.
        test_loader: DataLoader providing test data.
        threshold: The gradient threshold below which weights are masked (not updated).
    Returns:
        A dictionary of boolean masks for each parameter.
    """
    model.eval()
    # Initialize the mask dictionary
    mask = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            mask[name] = torch.zeros_like(param, dtype=torch.bool)
    
    # Iterate through the test data
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(next(model.parameters()).device), targets.to(next(model.parameters()).device)  # Move inputs to the model's device
        model.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # We minimize the norm of the logits (outputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        
        # Backward pass to compute gradients
        loss.backward()
        
        # Update the mask based on gradient magnitude
        for name, param in model.named_parameters():
            if param.grad is not None:
                mask[name] = mask[name] | ((param.data * param.grad.data).abs() > threshold)
    
    return mask

class CustomLoss(torch.nn.Module):
    def __init__(self, lambda_norm=1.0, lambda_ce=1.0, T=1.0):
        super(CustomLoss, self).__init__()
        self.lambda_norm = lambda_norm  # Weight for the logits norm penalty
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.lambda_ce = lambda_ce
        self.T = T

    def forward(self, logits, labels):
        # Split the logits and labels based on label conditions
        valid_idx = labels >= 0  # Mask for samples with labels >= 0
        invalid_idx = labels == -1  # Mask for samples with label -1

        # Cross entropy loss for valid labels (labels >= 0)
        if valid_idx.any():
            valid_logits = logits[valid_idx]
            valid_labels = labels[valid_idx]
            ce_loss = self.cross_entropy_loss(valid_logits, valid_labels)
        else:
            ce_loss = torch.tensor(0.0, device=logits.device)

        # Norm loss for invalid labels (label = -1)
        if invalid_idx.any():
            invalid_logits = logits[invalid_idx]
            norm_loss = torch.norm(invalid_logits, dim=1).mean()
            norm_loss = (invalid_logits/self.T).exp().sum(-1)
        else:
            norm_loss = torch.tensor(0.0, device=logits.device)

        # Combine the two losses
        total_loss = self.lambda_ce * ce_loss + self.lambda_norm * norm_loss
        return total_loss

@iterative_unlearn
def masked_energy_minimization(data_loaders, model, criterion, optimizer, epoch, args, mask=None, use_mask=True):

    T = 100

    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    forget_dataset = deepcopy(forget_loader.dataset)

    loss_fn = CustomLoss(T=args.temperature)

    if use_mask:
        mask = create_mask_from_gradients(model, forget_loader, threshold=args.mask_threshold)

    if args.dataset == "cifar100" or args.dataset == "TinyImagenet":
        try:
            forget_dataset.targets = np.random.ones(forget_dataset.targets.shape) * -1
        except:
            forget_dataset.dataset.targets = np.random.ones(len(forget_dataset.dataset.targets)) * -1
    
        retain_dataset = retain_loader.dataset
        train_dataset = torch.utils.data.ConcatDataset([forget_dataset,retain_dataset])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()

        # switch to train mode
        model.train()

        start = time.time()
        loader_len = len(forget_loader) + len(retain_loader)
        
        if epoch < args.warmup:
            utils.warmup_lr(epoch, i+1, optimizer,
                            one_epoch_step=loader_len, args=args)
            
        
        for it, (image, target) in enumerate(train_loader):
            i = it + len(forget_loader)
            image = image.cuda()
            target = target.cuda()
            output_clean = model(image)

            loss = loss_fn.forward(output_clean, target)
        
            optimizer.zero_grad()
            loss.backward()
            
            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
            
            optimizer.step()
        
            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]
        
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
      
    elif args.dataset == "cifar10" or args.dataset == "svhn":
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
      
        # switch to train mode
        model.train()
      
        start = time.time()
        loader_len = len(forget_loader) + len(retain_loader)
      
        if epoch < args.warmup:
            utils.warmup_lr(epoch, i+1, optimizer,
                            one_epoch_step=loader_len, args=args)
        
        for i, (image, target) in enumerate(forget_loader):
            image = image.cuda()
            target = torch.randint(0, args.num_classes, target.shape).cuda()
            
            # compute output
            output_clean = model(image)
            loss = loss_fn.forward(output_clean, target)
            
            optimizer.zero_grad()
            loss.backward()
            
            # print(mask)
            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
            
            optimizer.step()
            
        for i, (image, target) in enumerate(retain_loader):
            image = image.cuda()
            target = target.cuda()
            
            # compute output
            output_clean = model(image)
            loss = criterion(output_clean, target)
            
            optimizer.zero_grad()
            loss.backward()
            
            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
            
            optimizer.step()
            
            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]
            
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

    return top1.avg



