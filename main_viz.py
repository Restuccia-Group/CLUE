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
best_sa = 0

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

    def replace_loader_dataset(
        dataset, batch_size=args.batch_size, seed=1, shuffle=True
    ):
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

    criterion = nn.CrossEntropyLoss()
    evaluation_result = None

    if args.resume:
        checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)

    if args.resume and checkpoint is not None:
        model, evaluation_result = checkpoint
    else:
        if not args.chenyaofo:    
            checkpoint = torch.load(args.model_path, map_location=device)
            if "state_dict" in checkpoint.keys():
                checkpoint = checkpoint["state_dict"]

            if args.unlearn != "retrain":
                model.load_state_dict(checkpoint, strict=False)  


    # features, labels = extract_features(model, train_loader_full) 
    # print(f"The shape of features and labels are : {features.shape}, {labels.shape}")

    #Inspecting the ViT

    # for name, module in model.named_modules():
    #     print(name)
    #     print(module)

    #     print("\n")

    # raise Exception

    X,Y = get_tsne(model, train_loader_full)
    print(f"The shape of features and labels are : {X.shape}, {Y.shape}")

    np.savetxt(f"./viz_assets/{args.arch}_{args.dataset}_data.csv", X)
    np.savetxt(f"./viz_assets/{args.arch}_{args.dataset}_labels.csv", Y)


if __name__ == "__main__":
    main()