import sys
import time

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
        inputs = inputs.to(next(model.parameters()).device)  # Move inputs to the model's device
        model.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # We minimize the norm of the logits (outputs)
        loss = torch.nn.functional.cross_entropy(inputs, targets)
        
        # Backward pass to compute gradients
        loss.backward()
        
        # Update the mask based on gradient magnitude
        for name, param in model.named_parameters():
            if param.grad is not None:
                mask[name] = mask[name] | (param.grad.abs() > threshold)
    
    return mask


@iterative_unlearn
def masked_logit_minimization(data_loaders, model, criterion, optimizer, epoch, args, use_mask=True):
    train_loader = data_loaders["forget"]
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    if use_mask:
        mask = create_mask_from_gradients(model, train_loader, threshold=args.mask_threshold)

    # switch to train mode
    model.train()

    start = time.time()
    if args.imagenet_arch:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        for i, data in enumerate(train_loader):
            image, target = get_x_y_from_data_dict(data, device)

            # if epoch < args.warmup:
            #     utils.warmup_lr(
            #         epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
            #     )

            # compute output
            output_clean = model(image)
            loss = torch.norm(output_clean, p=2, dim=-1).mean()
            optimizer.zero_grad()
            loss.backward()

            if use_mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad.data *= mask[name].float()

            optimizer.step()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch, i, len(train_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()
    else:
        for i, (image, target) in enumerate(train_loader):
            # if epoch < args.warmup:
            #     utils.warmup_lr(
            #         epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
            #     )

            image = image.cuda()
            target = target.cuda()

            # compute output
            output_clean = model(image)
            loss = torch.norm(output_clean, p=2, dim=-1).mean()
            optimizer.zero_grad()
            loss.backward()

            if use_mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad.data *= mask[name].float()

            optimizer.step()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch, i, len(train_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg



