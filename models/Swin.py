import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm import create_model
import os



def swin_t(num_classes, img_size=32, patch_size=2, window_size=4):
    return create_model('swin_tiny_patch4_window7_224',
                     pretrained=True,
                     num_classes=num_classes,
                     img_size=img_size,
                     patch_size=patch_size, # Smaller patch size for smaller input
                     window_size=window_size) # Smaller window size for 32x32 inputs
    
def swin_s(num_classes, img_size=32, patch_size=2, window_size=4):
    return create_model('swin_small_patch4_window7_224',
                     pretrained=True,
                     num_classes=num_classes,
                     img_size=img_size,
                     patch_size=patch_size, # Smaller patch size for smaller input
                     window_size=window_size) # Smaller window size for 32x32 inputs

def swin_b(num_classes, img_size=32, patch_size=2, window_size=4):
    return create_model('swin_base_patch4_window7_224',
                     pretrained=True,
                     num_classes=num_classes,
                     img_size=img_size,
                     patch_size=patch_size, # Smaller patch size for smaller input
                     window_size=window_size) # Smaller window size for 32x32 inputs

def swin_l(num_classes, img_size=32, patch_size=2, window_size=4):
    return create_model('swin_large_patch4_window7_224',
                     pretrained=True,
                     num_classes=num_classes,
                     img_size=img_size,
                     patch_size=patch_size, # Smaller patch size for smaller input
                     window_size=window_size) # Smaller window size for 32x32 inputs

def vit_t(num_classes, img_size=32, patch_size=4):
    # Create Vision Transformer model
    return create_model('vit_tiny_patch16_224',
                        pretrained=True,
                        num_classes=num_classes,
                        img_size=32,  # Adjusting the image size for CIFAR-100
                        patch_size=4)  # Smaller patch size for smaller input images

def vit_b(num_classes, img_size=32, patch_size=4):
    # Create Vision Transformer model
    return create_model('vit_base_patch16_224',
                        pretrained=True,
                        num_classes=num_classes,
                        img_size=32,  # Adjusting the image size for CIFAR-100
                        patch_size=4)  # Smaller patch size for smaller input images

