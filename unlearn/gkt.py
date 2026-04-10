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


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
    

def attention(x):
        """
        Taken from https://github.com/szagoruyko/attention-transfer
        :param x = activations
        """
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


def attention_diff(x, y):
    """
    Taken from https://github.com/szagoruyko/attention-transfer
    :param x = activations
    :param y = activations
    """
    return (attention(x) - attention(y)).pow(2).mean()


def divergence(student_logits, teacher_logits, KL_temperature):
    divergence = F.kl_div(F.log_softmax(student_logits / KL_temperature, dim=1), F.softmax(teacher_logits / KL_temperature, dim=1))  # forward KL

    return divergence


def KT_loss_generator(student_logits, teacher_logits, KL_temperature):

    divergence_loss = divergence(student_logits, teacher_logits, KL_temperature)
    total_loss = -divergence_loss

    return total_loss


def KT_loss_student(student_logits, student_activations, teacher_logits, teacher_activations, KL_temperature = 1, AT_beta = 250):

    divergence_loss = divergence(student_logits, teacher_logits, KL_temperature)
    if AT_beta > 0:
        at_loss = 0
        for i in range(len(student_activations)):
            attention_loss = attention_diff(student_activations[i], teacher_activations[i])
            at_loss = at_loss + AT_beta * attention_loss
            # tqdm.write(f"{divergence_loss},{attention_loss}")
    else:
        at_loss = 0

    total_loss = divergence_loss + at_loss

    return total_loss

class Generator(nn.Module):

    def __init__(self, z_dim, out_size=32, num_channels = 3):
        super(Generator, self).__init__()
        inter_dim = z_dim // 2
        prefinal_layer = None
        final_layer = None
        if num_channels == 3:
            prefinal_layer = nn.Conv2d(inter_dim//2, 3, 3, stride=1, padding=1)
            final_layer = nn.BatchNorm2d(3, affine=True)
        elif num_channels == 1:
            prefinal_layer = nn.Conv2d(inter_dim//2, 1, 7, stride=1, padding=1)
            final_layer = nn.BatchNorm2d(1, affine=True)
        else:
            print(f"Generator Not Supported for {num_channels} channels")

        initial_size = out_size // 4  # We want to reach 4x4 spatial resolution
        initial_dim = inter_dim * initial_size * initial_size

        self.layers = nn.Sequential(
            nn.Linear(z_dim, initial_dim),
            nn.ReLU(inplace=True),
            View((-1, inter_dim, initial_size, initial_size)),
            nn.BatchNorm2d(inter_dim),

            nn.ConvTranspose2d(inter_dim, inter_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(inter_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(inter_dim, inter_dim//2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(inter_dim//2),
            nn.LeakyReLU(0.2, inplace=True),

            prefinal_layer,
            final_layer
        )


    def forward(self, z):
        return self.layers(z)

    def print_shape(self, x):
        """
        For debugging purposes
        """
        act = x
        for layer in self.layers:
            act = layer(act)
            print('\n', layer, '---->', act.shape)

            
class LearnableLoader(nn.Module):
    def __init__(self, n_repeat_batch, batch_size=256, z_dim=128, out_size=32, num_channels = 3,device='cuda'):
        """
        Infinite loader, which contains a learnable generator.
        """

        super(LearnableLoader, self).__init__()
        self.batch_size = batch_size
        self.n_repeat_batch = n_repeat_batch
        self.z_dim = z_dim
        self.generator = Generator(self.z_dim,out_size=out_size, num_channels=num_channels).to(device=device)
        self.device = device

        self._running_repeat_batch_idx = 0
        self.z = torch.randn((self.batch_size, self.z_dim)).to(device=self.device)

    def __next__(self):
        if self._running_repeat_batch_idx == self.n_repeat_batch:
            self.z = torch.randn((self.batch_size, self.z_dim)).to(device=self.device)
            self._running_repeat_batch_idx = 0

        images = self.generator(self.z)
        self._running_repeat_batch_idx += 1
        return images

    def samples(self, n, grid=True):
        """
        :return: if grid returns single grid image, else
        returns n images.
        """
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn((n, self.z_dim)).to(device=self.device)
            images = visualize(self.generator(z), dataset=self.dataset).cpu()
            if grid:
                images = torchvision.utils.make_grid(images, nrow=round(math.sqrt(n)), normalize=True)

        self.generator.train()
        return images

    def __iter__(self):
        return self

def gkt_unlearn(data_loaders, model, criterion, optimizer, epoch, args, mask=None, use_mask=True):
    student = model

    #Default configuration
    KL_temperature = 1
    AT_beta = 250
    n_generator_iter = 1
    n_student_iter = 10
    n_repeat_batch = n_generator_iter + n_student_iter


    generator = LearnableLoader(n_repeat_batch=n_repeat_batch, num_channels = 3, device = args.device).to(device=args.device)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=0.001) 
    scheduler_generator = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_generator, 
                                                                mode='min', factor=0.5, patience=2, verbose=True)
    
    # saving the generator
    if not os.path.exists("../saved_models/generator_", str(0) + ".pt"):
        torch.save(generator.state_dict(), os.path.join("../saved_models/generator_gkt/generator_", str(0) + ".pt"))

    #optimizer_student = torch.optim.Adam(student.parameters(), lr=0.001)
    scheduler_student = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, \
                                        mode='min', factor=0.5, patience=2, verbose=True)
    
    idx_pseudo = 0
    total_n_pseudo_batches = 4000
    n_pseudo_batches = 0
    running_gen_loss = []
    running_stu_loss = []
    threshold = 0.01

    # Performing Unlearnign
    while n_pseudo_batches < total_n_pseudo_batches:
        x_pseudo = generator.__next__()
        preds, *_ = model(x_pseudo)
        mask = (torch.softmax(preds.detach(), dim=1)[:, 0] <= threshold)
        x_pseudo = x_pseudo[mask]
        if x_pseudo.size(0) == 0:
            zero_count += 1
            if zero_count > 100:
                print("Generator Stopped Producing datapoints corresponding to retain classes.")
                print("Resetting the generator to previous checkpoint")
                generator.load_state_dict(torch.load(os.path.join("../saved_models/generator_gkt/generator_", str(((n_pseudo_batches//50)-1)*50) + ".pt")))
            continue
        else:
            zero_count = 0
        
        ## Take n_generator_iter steps on generator
        if idx_pseudo % n_repeat_batch < n_generator_iter:
            student_logits, *student_activations = student(x_pseudo)
            teacher_logits, *teacher_activations = model(x_pseudo)
            generator_total_loss = KT_loss_generator(student_logits, teacher_logits, KL_temperature=KL_temperature)

            optimizer_generator.zero_grad()
            generator_total_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 5)
            optimizer_generator.step()
            running_gen_loss.append(generator_total_loss.cpu().detach())


        elif idx_pseudo % n_repeat_batch < (n_generator_iter + n_student_iter):
            
            
            with torch.no_grad():
                teacher_logits, *teacher_activations = model(x_pseudo)

            student_logits, *student_activations = student(x_pseudo)
            student_total_loss = KT_loss_student(student_logits, student_activations, 
                                                teacher_logits, teacher_activations, 
                                                KL_temperature=KL_temperature, AT_beta = AT_beta)

            optimizer.zero_grad()
            student_total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5)
            optimizer.step()
            running_stu_loss.append(student_total_loss.cpu().detach())
            
# @iterative_unlearn
# def masked_energy_minimization(data_loaders, model, criterion, optimizer, epoch, args, mask=None, use_mask=True):

#     T = 100

#     forget_loader = data_loaders["forget"]
#     retain_loader = data_loaders["retain"]
#     forget_dataset = deepcopy(forget_loader.dataset)

#     loss_fn = CustomLoss(T=args.temperature)

#     if use_mask:
#         mask = create_mask_from_gradients(model, forget_loader, threshold=args.mask_threshold)

#     if args.dataset == "cifar100" or args.dataset == "TinyImagenet":
#         try:
#             forget_dataset.targets = np.random.ones(forget_dataset.targets.shape) * -1
#         except:
#             forget_dataset.dataset.targets = np.random.ones(len(forget_dataset.dataset.targets)) * -1
    
#         retain_dataset = retain_loader.dataset
#         train_dataset = torch.utils.data.ConcatDataset([forget_dataset,retain_dataset])
#         train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
#         losses = utils.AverageMeter()
#         top1 = utils.AverageMeter()

#         # switch to train mode
#         model.train()

#         start = time.time()
#         loader_len = len(forget_loader) + len(retain_loader)
        
#         if epoch < args.warmup:
#             utils.warmup_lr(epoch, i+1, optimizer,
#                             one_epoch_step=loader_len, args=args)
            
        
#         for it, (image, target) in enumerate(train_loader):
#             i = it + len(forget_loader)
#             image = image.cuda()
#             target = target.cuda()
#             output_clean = model(image)

#             loss = loss_fn.forward(output_clean, target)
        
#             optimizer.zero_grad()
#             loss.backward()
            
#             if mask:
#                 for name, param in model.named_parameters():
#                     if param.grad is not None:
#                         param.grad *= mask[name]
            
#             optimizer.step()
        
#             output = output_clean.float()
#             loss = loss.float()
#             # measure accuracy and record loss
#             prec1 = utils.accuracy(output.data, target)[0]
        
#             losses.update(loss.item(), image.size(0))
#             top1.update(prec1.item(), image.size(0))
        
#             if (i + 1) % args.print_freq == 0:
#                 end = time.time()
#                 print('Epoch: [{0}][{1}/{2}]\t'
#                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                         'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
#                         'Time {3:.2f}'.format(
#                             epoch, i, loader_len, end-start, loss=losses, top1=top1))
#                 start = time.time()
      
#     elif args.dataset == "cifar10" or args.dataset == "svhn":
#         losses = utils.AverageMeter()
#         top1 = utils.AverageMeter()
      
#         # switch to train mode
#         model.train()
      
#         start = time.time()
#         loader_len = len(forget_loader) + len(retain_loader)
      
#         if epoch < args.warmup:
#             utils.warmup_lr(epoch, i+1, optimizer,
#                             one_epoch_step=loader_len, args=args)
        
#         for i, (image, target) in enumerate(forget_loader):
#             image = image.cuda()
#             target = torch.randint(0, args.num_classes, target.shape).cuda()
            
#             # compute output
#             output_clean = model(image)
#             loss = loss_fn.forward(output_clean, target)
            
#             optimizer.zero_grad()
#             loss.backward()
            
#             # print(mask)
#             if mask:
#                 for name, param in model.named_parameters():
#                     if param.grad is not None:
#                         param.grad *= mask[name]
            
#             optimizer.step()
            
#         for i, (image, target) in enumerate(retain_loader):
#             image = image.cuda()
#             target = target.cuda()
            
#             # compute output
#             output_clean = model(image)
#             loss = criterion(output_clean, target)
            
#             optimizer.zero_grad()
#             loss.backward()
            
#             if mask:
#                 for name, param in model.named_parameters():
#                     if param.grad is not None:
#                         param.grad *= mask[name]
            
#             optimizer.step()
            
#             output = output_clean.float()
#             loss = loss.float()
#             # measure accuracy and record loss
#             prec1 = utils.accuracy(output.data, target)[0]
            
#             losses.update(loss.item(), image.size(0))
#             top1.update(prec1.item(), image.size(0))
            
#             if (i + 1) % args.print_freq == 0:
#                end = time.time()
#                print('Epoch: [{0}][{1}/{2}]\t'
#                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                      'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
#                      'Time {3:.2f}'.format(
#                          epoch, i, loader_len, end-start, loss=losses, top1=top1))
#                start = time.time()

#     return top1.avg



