import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm import create_model
import os
import timm

print(timm.__version__)

# raise Exception

# def vit_b(num_classes, img_size=32, patch_size=4):
#     # Create Vision Transformer model
#     return create_model('vit_base_patch16_224',
#                         pretrained=True,
#                         num_classes=num_classes,
#                         img_size=32,  # Adjusting the image size for CIFAR-100
#                         patch_size=4)  # Smaller patch size for smaller input images


# model = vit_b(100)

# path = "/home/sazzad/Machine Unlearning/Unlearn-Saliency-master/Classification/saved_model/vit_b_cifar100_model_SA_best.pth.tar"

# model_weights = torch.load(path)

# del(model_weights['state_dict']['normalize.mean'])
# del(model_weights['state_dict']['normalize.std'])

# model.load_state_dict(model_weights['state_dict'])

# print(model_weights['result'])
# print(model_weights['best_sa'])