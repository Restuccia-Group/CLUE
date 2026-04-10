import sys
import time
import numpy as np
from copy import deepcopy
import torch.multiprocessing as mp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils

from .impl import iterative_unlearn

sys.path.append(".")
from imagenet import get_x_y_from_data_dict

# Creates a mask from the forget set
def create_mask_from_gradients(model, test_loader, threshold=85):
    """
    Creates a mask for the weights of a model based on gradient information from the test data.
    Args:
        model: The neural network model.
        test_loader: DataLoader providing test data.
        percentile: The percentile of gradient values in each layer to determine the threshold.
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
        
        # Update the mask based on the layer-wise percentile of gradient values
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Compute Taylor criterion
                taylor_criterion = (param.data * param.grad.data).abs()
                
                # Determine the threshold based on the percentile
                # layer_threshold = torch.quantile(taylor_criterion.view(-1), threshold)
                
                # Update the mask with the computed threshold
                mask[name] = mask[name] | (taylor_criterion > threshold)
    # count = 0
    # num_el = 0
    # for name in mask:
    #     count += mask[name].count_nonzero().item()
    #     num_el += torch.numel(mask[name])

    # print("the quantile of the threshold:", count, num_el)

    
    return mask
    
def add_gaussian_noise(images, std_devs):
    """
    Adds Gaussian noise to a batch of images with varying standard deviations.
    
    Args:
        images (torch.Tensor): A batch of images of shape (batch_size, channels, height, width).
        std_devs (torch.Tensor): A tensor of shape (batch_size,) containing the standard deviation 
                                 of the Gaussian noise for each image in the batch.
    
    Returns:
        torch.Tensor: A batch of images with Gaussian noise added.
    """
    # Ensure std_devs is the same device as images and reshape it for broadcasting
    std_devs = std_devs.view(-1, 1, 1, 1).to(images.device)
    
    # Generate noise with the same shape as images, with varying std_devs
    noise = torch.randn_like(images) * std_devs
    
    # Add noise to the images
    noisy_images = images + noise
    
    return noisy_images.clip(min=0, max=1)

# Function to clone the weights from a pretrained model
def clone_weights(source_net, target_net):
    target_net.load_state_dict(source_net.state_dict())


def add_salt_and_pepper_noise_batch(images, salt_prob=0.01, pepper_prob=0.01):
    # Create a copy of the original batch
    noisy_images = images.clone()

    # Iterate through each image in the batch
    for i in range(images.shape[0]):
        # Get the current image
        image = images[i]

        # Get the total number of pixels
        num_pixels = image.numel()

        # Generate random numbers for salt
        num_salt = int(num_pixels * salt_prob)
        coords = [np.random.randint(0, j - 1, num_salt) for j in image.shape]
        noisy_images[i, coords[0], coords[1]] = 1  # Set to white (salt)

        # Generate random numbers for pepper
        num_pepper = int(num_pixels * pepper_prob)
        coords = [np.random.randint(0, j - 1, num_pepper) for j in image.shape]
        noisy_images[i, coords[0], coords[1]] = 0  # Set to black (pepper)

    return noisy_images

def generate_softmax_with_zero(batch_size, num_classes, zero_class_idx):
    # Step 1: Generate random values uniformly between [1, 10]
    labels = torch.rand(batch_size, num_classes) * 9 + 1  # Scales values to [1, 10]

    # Step 2: Set the specified class to zero
    labels[:, zero_class_idx] = 0

    # Step 3: Normalize so that probabilities sum to 1
    labels /= labels.sum(dim=-1, keepdim=True)
    labels[:, zero_class_idx] = 1e-5

    return labels

def ood_assisted_unlearning(model, train_loader, mask, optimizer, losses, top1, epoch, loader_len, args):
    # Copy the pretrained model for both teacher and student

    teacher = deepcopy(model)######################
    # student = model

    # Since both teacher and student start with the same pretrained weights, clone weights
    # clone_weights(teacher, student)

    # Freeze the teacher network
    for param in teacher.parameters():###############################
        param.requires_grad = False###################################

    teacher.eval()##########################
    model.train()

    # Define optimizer for the student network
    # optimizer = optim.Adam(model.parameters(), lr=args.unlearn_lr)

    # Loss function (for matching teacher output)
    # loss_fn = nn.MSELoss() 
    loss_fn = nn.KLDivLoss(log_target=True, reduction="batchmean")

    start = time.time()

    for it, (image, target) in enumerate(train_loader):
        # i = len(train_loader)
        image = image.cuda()
        target = target.cuda()

        # Random noise strengths for each sample
        # Generate random standard deviations between 0.1 and 0.5 for each image in the batch
        batch_size = image.size(0)
        std_devs = torch.rand(batch_size) * 0.5 + 0.5  # Random values between 0.5 and 1.0
        noisy_input = add_gaussian_noise(image, std_devs)   #Are we considering mean and std dev?

        # noisy_input = add_salt_and_pepper_noise_batch(image, salt_prob=0.5, pepper_prob=0.5)    #############################################

        # Forward pass through the teacher network
        with torch.no_grad():#####################################################
            teacher_output = teacher(noisy_input)########################################################

        # Add noise to input and pass through the student network
        
        student_output = model(image)

        # print(student_output.max().item(), student_output.min().item())

        # Compute the loss between the teacher's and student's outputs
        
        loss = (student_output/student_output.norm(2, dim=-1).view(-1, 1)).exp().sum(-1).mean() + loss_fn(nn.functional.log_softmax(teacher_output, dim=-1), nn.functional.log_softmax(student_output, dim=-1))#######################################
        # loss = loss_fn(generate_softmax_with_zero(student_output.shape[0], student_output.shape[1], target[0].item()).log().cuda(), nn.functional.log_softmax(student_output, dim=-1))
        


        #((student_output).max(dim=-1)[0] - (student_output).min(dim=-1)[0]).mean(-1) +
        # Backpropagation on the student network
        optimizer.zero_grad()
        loss.backward()
        
        if mask:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad *= mask[name]
        
        optimizer.step()

        # print(f"Peak Memory Allocated: {torch.cuda.max_memory_allocated()/(1024**2)} MB")

        # raise Exception
    
        output = student_output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = utils.accuracy(output.data, target)[0]
    
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))
    
        if (it + 1) % args.print_freq == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Time {3:.2f}'.format(
                        epoch, it, loader_len, end-start, loss=losses, top1=top1))
            start = time.time()

    # return student



@iterative_unlearn
def ood_unlearning(data_loaders, model, criterion, optimizer, epoch, args, mask=None, use_mask=True):

    forget_loader = data_loaders["forget"]
    train_dataset = deepcopy(forget_loader.dataset)
    loader_len = len(forget_loader)

    if use_mask:
        mask = create_mask_from_gradients(model, forget_loader, threshold=args.mask_threshold)

    if args.dataset == "cifar100" or args.dataset == "TinyImagenet":
        # try:
        #     train_dataset.targets = np.random.ones(train_dataset.targets.shape) * -1
        # except:
        #     train_dataset.dataset.targets = np.random.ones(len(train_dataset.dataset.targets)) * -1
    
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()

        # switch to train mode
        model.train()

        start = time.time()
            
        ood_assisted_unlearning(model, train_loader, mask, optimizer, losses, top1, epoch, loader_len, args)
      
    elif args.dataset == "cifar10" or args.dataset == "svhn":
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
      
        # switch to train mode
        model.train()
      
        start = time.time()
        loader_len = len(forget_loader)

        ood_assisted_unlearning(model, train_loader, mask, optimizer, losses, top1, epoch, loader_len, args)
      
    return top1.avg
