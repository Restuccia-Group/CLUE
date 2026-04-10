import torch
import sys
import os

num_classes = 10
code = f"cifar{num_classes}"
model_name = "resnet"
model_version = "56"

model = torch.hub.load("chenyaofo/pytorch-cifar-models", "{code}_{model_name}(model_version)", pretrained=True)