import torchvision
import torch

def vit_b_16(num_classes=None):
    return torchvision.models.vit_b_16(weights="IMAGENET1K_V1")

def vit_b_32(num_classes=None):
    return torchvision.models.vit_b_32(weights="IMAGENET1K_V1")

def vit_l_16(num_classes=None):
    return torchvision.models.vit_l_16(weights="IMAGENET1K_V1")

def vit_l_32(num_classes=None):
    return torchvision.models.vit_l_32(weights="IMAGENET1K_V1")

def vit_h_14(num_classes=None):
    return torchvision.models.vit_h_14(weights="IMAGENET1K_SWAG_E2E_V1")

def swin_v2_t(num_classes=None):
    return torchvision.models.swin_v2_t(weights="IMAGENET1K_V1")

def swin_v2_s(num_classes=None):
    return torchvision.models.swin_v2_s(weights="IMAGENET1K_V1")

def swin_v2_b(num_classes=None):
    return torchvision.models.swin_v2_b(weights="IMAGENET1K_V1")

