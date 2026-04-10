import torch
import torch.nn as nn
from tqdm import tqdm


import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm import create_model
import os
import timm
import argparse
import os
import pdb
import pickle
import random
import shutil
import time
import copy
from copy import deepcopy
from collections import OrderedDict

import arg_parser
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from trainer import train, validate
import utils
import unlearn

from viz_utils.tsne import extract_features, get_tsne
from viz_utils.vit_explain import attention_rollout

from sklearn.feature_selection import mutual_info_classif

def reduce_data(data_set, percentage, seed):
    valid_idx = []
    rng = np.random.RandomState(seed)
    for i in range(max(data_set.targets) + 1):
        class_idx = np.where(data_set.targets == i)[0]
        valid_idx.append(
            rng.choice(class_idx, int(percentage * len(class_idx)), replace=False)
        )
    valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(data_set)

    data_set.data = train_set_copy.data[valid_idx]
    data_set.targets = train_set_copy.targets[valid_idx]
    return data_set

def measure_class_information(model_scrubf, modelf0, delta_w_s, delta_w_m0, 
                              data_loader, loss_fn=nn.CrossEntropyLoss()):
    """
    Measures how much information about a specific class is retained in activations after forgetting.
    
    Args:
        model_scrubf: The model after forgetting (scrubbed model).
        modelf0: The original model before forgetting.
        delta_w_s: Perturbation in scrubbed model parameters.
        delta_w_m0: Perturbation in original model parameters.
        data_loader: DataLoader containing only samples from the target class.
        loss_fn: Loss function (default: CrossEntropy).
    
    Returns:
        A dictionary containing average KL divergence and loss metrics.
    """
    model_scrubf.eval()
    modelf0.eval()
    
    data_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size=1, shuffle=False)
    
    kl_divergence_sum = 0.0
    sample_count = 0
    
    for batch in tqdm(data_loader, leave=False, desc="Processing Target Class"):
        batch = [tensor.to(next(model_scrubf.parameters()).device) for tensor in batch]
        input, target = batch
        
        sample_count += 1

        # Forward pass through both models
        output_sf = model_scrubf(input)
        output_m0 = modelf0(input)

        # Get target class index from output (assuming one-hot encoding or softmax logits)
        target_class = target.item()

        # Compute gradients for the specific class
        grads_sf = torch.autograd.grad(output_sf[0, target_class], model_scrubf.parameters(), retain_graph=False)
        grads_m0 = torch.autograd.grad(output_m0[0, target_class], modelf0.parameters(), retain_graph=False)
        
        # Flatten gradients and compute sensitivity measures
        G_sf = torch.cat([g.view(-1) for g in grads_sf]).pow(2)
        G_m0 = torch.cat([g.view(-1) for g in grads_m0]).pow(2)
        
        delta_f_sf = torch.matmul(G_sf, delta_w_s)
        delta_f_m0 = torch.matmul(G_m0, delta_w_m0)

        # KL divergence-based forgetting measure
        kl = ((output_m0[0, target_class] - output_sf[0, target_class]).pow(2) / delta_f_m0 
              + delta_f_sf / delta_f_m0 - torch.log(delta_f_sf / delta_f_m0) - 1)

        kl_divergence_sum += kl.item()
    
    if sample_count == 0:
        return {"error": "No samples found in the dataloader."}

    # Compute the average KL divergence over all target class samples
    avg_kl_div = kl_divergence_sum / sample_count

    return {"avg_kl_divergence": avg_kl_div, "samples_used": sample_count}



def vit_b(num_classes, img_size=32, patch_size=4):
    # Create Vision Transformer model
    return create_model('vit_base_patch16_224',
                        pretrained=True,
                        num_classes=num_classes,
                        img_size=32,  # Adjusting the image size for CIFAR-100
                        patch_size=4)  # Smaller patch size for smaller input images


###########################
def main():
    args = arg_parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed
    # prepare dataset
    (
        model,
        train_loader_full,
        val_loader,
        test_loader,
        marked_loader,
    ) = utils.setup_model_dataset(args)
    model.cuda()
    # print("After loading datasets")
    # backup_train_full_loader = deepcopy(train_loader_full)

    def replace_loader_dataset(
        dataset, batch_size=args.batch_size, seed=1, shuffle=True
    ):
        # print("In replace loader")
        utils.setup_seed(seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=shuffle,
        )

    forget_dataset = copy.deepcopy(marked_loader.dataset)
    if args.dataset == "svhn":
        try:
            marked = forget_dataset.targets < 0
        except:
            marked = forget_dataset.labels < 0
        forget_dataset.data = forget_dataset.data[marked]
        try:
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
        except:
            forget_dataset.labels = -forget_dataset.labels[marked] - 1
        forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
        retain_dataset = copy.deepcopy(marked_loader.dataset)
        try:
            marked = retain_dataset.targets >= 0
        except:
            marked = retain_dataset.labels >= 0
        retain_dataset.data = retain_dataset.data[marked]
        try:
            retain_dataset.targets = retain_dataset.targets[marked]
        except:
            retain_dataset.labels = retain_dataset.labels[marked]
        retain_dataset = reduce_data(retain_dataset, args.retain_percentage, seed)  #This will reduce the size of the retain set availabel to the unlearning algorithm
        retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)
        assert len(forget_dataset) + args.retain_percentage * len(retain_dataset) == len(
            train_loader_full.dataset
        )
    else:
        try:
            marked = forget_dataset.targets < 0
            forget_dataset.data = forget_dataset.data[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=True
            )
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.data = retain_dataset.data[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_dataset = reduce_data(retain_dataset, args.retain_percentage, seed)  #This will reduce the size of the retain set availabel to the unlearning algorithm
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True
            )
            # assert len(forget_dataset) + args.retain_percentage * len(retain_dataset) == len(
            #     train_loader_full.dataset
            # )
        except:
            marked = forget_dataset.targets < 0
            forget_dataset.imgs = forget_dataset.imgs[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=True
            )
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.imgs = retain_dataset.imgs[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_dataset = reduce_data(retain_dataset, args.retain_percentage, seed)  #This will reduce the size of the retain set availabel to the unlearning algorithm
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True
            )
            assert len(forget_dataset) + args.retain_percentage * len(retain_dataset) == len(
                train_loader_full.dataset
            )

    print(f"number of retain dataset {len(retain_dataset)}")
    print(f"number of forget dataset {len(forget_dataset)}")
    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
    )

    return unlearn_data_loaders

#### Loading the Original Model ####
if __name__ == "__main__":
    ul_loader = main()
    model_orig = vit_b(10)

    path = "/home/sazzad/Machine Unlearning/Unlearn-Saliency-master/Classification/saved_model/vit_b_cifar10_checkpoint.pth.tar"

    model_weights = torch.load(path)

    del(model_weights['state_dict']['normalize.mean'])
    del(model_weights['state_dict']['normalize.std'])

    model_orig.load_state_dict(model_weights['state_dict'])

    #### Loading the forgotten model ####
    model_ul = vit_b(10)

    path = "/home/sazzad/Machine Unlearning/Unlearn-Saliency-master/Classification/results/oodcheckpoint.pth.tar"

    model_weights = torch.load(path)

    del(model_weights['state_dict']['normalize.mean'])
    del(model_weights['state_dict']['normalize.std'])

    model_ul.load_state_dict(model_weights['state_dict'])


    forget_loader = ul_loader['forget']
    retain_loader = ul_loader['retain']

    # correct = 0
    # samples = 0

    # model_orig.cuda()
    # model_ul.cuda()
    # with torch.no_grad():
    #     for x,y in forget_loader:
    #         x = x.cuda()
    #         y = y.cuda()

    #         output = model_ul(x)
    #         output = output.argmax(-1, keepdims=True)
    #         correct += output.eq(y.view_as(output)).sum().item()
    #         samples += len(y)

    # print("Accuacy:", correct/samples)
    # Target class
    TARGET_CLASS = 2

    # Hook function to extract activations from last few blocks
    activations = []
    def hook_fn(module, input, output):
        activations.append(output[:, 0, :].squeeze().detach().cpu().numpy())

    # Attach hooks to last 3 transformer blocks
    num_blocks = len(model_orig.blocks)  # Number of blocks in ViT
    for i in range(-3, 0):  # Last 3 blocks
        model_ul.blocks[i].register_forward_hook(hook_fn)

    # Collect activations and labels
    act_list, label_list = [], []

    model_orig.eval()
    model_ul.eval()
    with torch.no_grad():
        for imgs, labels in forget_loader:
            # if (labels == TARGET_CLASS).sum() == 0:
            #     continue  # Skip batches without target class

            activations.clear()  # Reset activations list

            # Forward pass
            outputs = model_ul(imgs)

            outputs = outputs.argmax(-1)

            # Store activations for target class images
            # print(np.array(activations).shape)
            activations_ = np.array(activations)
            # print(activations_.shape)
            activations_ = np.moveaxis(activations, 0, 1)
            act_list.append(activations_.reshape(activations_.shape[0], -1))
            label_list.append(outputs.cpu().numpy().reshape(-1, 1))
            # for act, label in zip(activations_, labels.numpy()):
            #     if label == TARGET_CLASS:
            #         act_list.append(act.flatten())  # Flatten activation map
            #         label_list.append(label)

    # Convert to numpy arrays
    X = np.concatenate(act_list, axis=0)  # Activations

    print(X.shape)
    Y = np.concatenate(label_list, axis=0)  # Labels (all 2s)

    # Compute mutual information
    mi_score = mutual_info_classif(X, Y.ravel(), discrete_features=False)
    print(f"Mutual Information Score for class {TARGET_CLASS}: {np.mean(mi_score):.4f}")






