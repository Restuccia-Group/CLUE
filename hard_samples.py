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

def vit_b(num_classes, img_size=32, patch_size=4):
    # Create Vision Transformer model
    return create_model('vit_base_patch16_224',
                        pretrained=True,
                        num_classes=num_classes,
                        img_size=32,  # Adjusting the image size for CIFAR-100
                        patch_size=4)  # Smaller patch size for smaller input images


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



def distance_to_decision_boundaries(x, y, model=None):
    """
    Calculates the distance of each input to the decision boundaries of a DNN model using a for loop.

    Args:
        model (torch.nn.Module): Trained DNN model.
        x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

    Returns:
        distances (torch.Tensor): Tensor of shape (batch_size, num_classes - 1), containing distances to decision boundaries.
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # Get the logits (z)
    with torch.no_grad():
        z = model(x)  # Shape: (batch_size, num_classes)

    # Identify the predicted class indices
    pred_classes = torch.argmax(z, dim=1)  # Shape: (batch_size,)

    # Extract the weights of the final layer
    # Assumes the final layer is named 'fc' or equivalent
    if hasattr(model, "fc") and isinstance(model.fc, torch.nn.Linear):
        weight_vectors = model.fc.weight  # Shape: (num_classes, feature_dim)
    else:
        weight_vectors = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                weight_vectors = module.weight

    distances = []
    nearest_classes = []
    for i in range(z.size(0)):  # Iterate over the batch
        pred_class = pred_classes[i].item()
        z_pred = z[i, pred_class].item()

        distances_per_sample = []
        classes_per_sample = []
        nearest_class = None
        min_dist = np.inf

        for j in range(z.size(1)):  # Iterate over all classes
            if j==pred_class:
                # distances_per_sample.append(-10)
                continue

            z_i = z[i, j].item()
            w_pred = weight_vectors[pred_class]
            w_i = weight_vectors[j]

            # Calculate the L2 norm of the difference between weight vectors
            norm_diff = torch.norm(w_pred - w_i, p=2).item()

            # Calculate the distance to the decision boundary for class j
            distance = np.abs(z_pred - z_i) / norm_diff

            distances_per_sample.append(distance)
            classes_per_sample.append(j)

        distances.append(min(distances_per_sample))
        nearest_classes.append(classes_per_sample[distances_per_sample.index(min(distances_per_sample))])

    # distances = np.array(distances)

    return np.array(distances), np.array(nearest_classes)



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

### Loading the Original Model ####
# if __name__ == "__main__":
#     ul_loader = main()
#     model_orig = vit_b(100)

#     path = "/home/sazzad/Machine Unlearning/Unlearn-Saliency-master/Classification/saved_model/vit_b_cifar100_checkpoint.pth.tar"

#     model_weights = torch.load(path)

#     del(model_weights['state_dict']['normalize.mean'])
#     del(model_weights['state_dict']['normalize.std'])

#     model_orig.load_state_dict(model_weights['state_dict'])

#     #### Loading the forgotten model ####
#     model_ul = vit_b(100)

#     path = "/home/sazzad/Machine Unlearning/Unlearn-Saliency-master/Classification/results/oodcheckpoint.pth.tar"

#     model_weights = torch.load(path)

#     del(model_weights['state_dict']['normalize.mean'])
#     del(model_weights['state_dict']['normalize.std'])

#     model_ul.load_state_dict(model_weights['state_dict'])


#     forget_loader = ul_loader['forget']
#     retain_loader = ul_loader['retain']

#     distances = []
#     prev_pred = []
#     ul_pred = []
#     nearest_classes = []

#     gt = []

#     model_ul.eval()
#     model_orig.eval()
#     with torch.no_grad():
#         for imgs, labels in forget_loader:
#             gt.extend(list(labels.cpu().numpy().ravel()))

#             # Forward pass
#             outputs_orig = model_orig(imgs)
#             outputs_ul = model_ul(imgs)

#             batch_distances, batch_classes = distance_to_decision_boundaries(imgs, labels, model=model_orig)
#             distances.extend(list(batch_distances))
#             nearest_classes.extend(list(batch_classes))


#             outputs_orig = outputs_orig.argmax(-1)
#             outputs_ul = outputs_ul.argmax(-1)

#             prev_pred.extend(list(outputs_orig.cpu().numpy().ravel()))
#             ul_pred.extend(list(outputs_ul.cpu().numpy().ravel()))

#     distances = np.array(distances)
#     prev_pred = np.array(prev_pred)
#     ul_pred = np.array(ul_pred)
#     nearest_classes = np.array(nearest_classes)

#     path = "/home/sazzad/Machine Unlearning/Unlearn-Saliency-master/Classification/rebuttal_results/vit_b_cifar100_forget_"
#     np.save(path + "distances.npy", distances)
#     np.save(path + "orig_pred.npy", prev_pred)
#     np.save(path + "ul_pred.npy", ul_pred)
#     np.save(path + "gt.npy", np.array(gt))
#     np.save(path + "nearest_classes.npy", nearest_classes)



######Checking label change
path = "/home/sazzad/Machine Unlearning/Unlearn-Saliency-master/Classification/rebuttal_results/"
distance = np.load(path + "distances.npy")
orig_pred = np.load(path + "orig_pred.npy")
ul_pred = np.load(path + "ul_pred.npy")
gt = np.load(path + "gt.npy")


############## Here, we will check for each dist bin what percentage is misclassified/corrected #########

samples_misclassified_after = 1 - (ul_pred.ravel()==gt.ravel())
samples_misclassified_before = 1 - (orig_pred.ravel()==gt.ravel())

#which samples are correctly classified
samples_correct_after = (ul_pred.ravel()==gt.ravel())
samples_correct_before = (orig_pred.ravel()==gt.ravel())

change_in_pred = 1 - (orig_pred.ravel()==ul_pred.ravel())

samples_misclassified_for_ul = np.clip(samples_misclassified_after - samples_misclassified_before, a_max=1, a_min=0) * change_in_pred
samples_corrected_for_ul = np.clip( - samples_misclassified_after + samples_misclassified_before, a_max=1, a_min=0) * change_in_pred


sorted_ind = np.argsort(distance)
change_in_pred = change_in_pred[sorted_ind]
samples_misclassified_after = samples_misclassified_after[sorted_ind]
samples_misclassified_before = samples_misclassified_before[sorted_ind]

samples_misclassified_for_ul = samples_misclassified_for_ul[sorted_ind]
samples_corrected_for_ul = samples_corrected_for_ul[sorted_ind]


# samples_misclassified = [1 if (x == 1 and y == 0) else 0 for x, y in zip(samples_misclassified_before, samples_misclassified_after)]

# samples_misclassified = np.array(samples_misclassified).astype(int)

change_in_pred_cumul = np.cumsum(change_in_pred)
distance = distance[sorted_ind]

import matplotlib.pyplot as plt

# Define bins (e.g., 10 equal-width bins between min and max distances)
bins = np.linspace(np.min(distance), np.max(distance), 11)

# Digitize distances into bins
binned_indices = np.digitize(distance, bins)

# Count label changes in each bin
bin_counts_misclassified = [np.sum(samples_misclassified_for_ul[binned_indices == i]) for i in range(1, len(bins))] #Checking how many samples are misclassified in this bin
bin_counts_correct = [np.sum(samples_corrected_for_ul[binned_indices == i]) for i in range(1, len(bins))]
bin_counts_changed = [np.sum(change_in_pred[binned_indices == i]) for i in range(1, len(bins))]
bin_counts_unchanged = [np.sum((1-change_in_pred)[binned_indices == i]) for i in range(1, len(bins))]

print("Changed:", bin_counts_changed)
print("Correct:", bin_counts_correct)
print("Misclassified:", bin_counts_misclassified)

print("Unchanged:", bin_counts_unchanged)

# raise

print(len(bin_counts_correct), len(bin_counts_changed))
bin_counts = np.array(bin_counts_misclassified) / (np.array(bin_counts_changed) + 1e-12)# + np.array(bin_counts_misclassified) + np.array(bin_counts_correct))


# Plot the bar chart
plt.figure(figsize=(8, 6))
plt.bar(bins[:-1], 100 * (bin_counts), width=np.diff(bins), edgecolor="black", align="edge", color="skyblue")

# Set labels and font size
plt.xlabel('Distance', fontsize=40)
plt.ylabel('Frequency of \nSamples (in %)', fontsize=40)

# Select only 5 evenly spaced ticks
bin_centers = (bins[:-1] + bins[1:]) / 2
tick_indices = np.linspace(0, len(bin_centers) - 1, 3, dtype=int)  # Select 5 indices
# y_tick_indices = np.linspace(0, 60 - 1, 6, dtype=int)  # Select 5 indices
plt.xticks(bin_centers[tick_indices], fontsize=40)
# plt.yticks(y_tick_indices, fontsize=40)

# Adjust tick label font size and layout
plt.yticks(fontsize=40)
plt.tight_layout()  # Ensures labels and ticks are not cut off

# Add grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

# Compute the center of each bin for the x-axis For Line graph
# bin_centers = (bins[:-1] + bins[1:]) / 2

# # Plot the line plot
# plt.figure(figsize=(7, 6))
# plt.plot(bin_centers, 100 * (bin_counts/np.sum(bin_counts)), marker='o', linestyle='-', color='b', label='Misclassification Rate')

# # Set labels and font size
# plt.xlabel('Distance Bins', fontsize=20)
# plt.ylabel('Frequency of Changed Sampled', fontsize=20)

# # Adjust tick label font size
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)

# # Customize legend
# plt.legend(fontsize=20)

# # Add grid for better readability
# plt.grid(axis='both', linestyle='--', alpha=0.7)

# plt.show()



# print((len(orig_pred) - np.sum(change_in_pred))/len(orig_pred))









################### Checking for heirrarchical classes

# path = "/home/sazzad/Machine Unlearning/Unlearn-Saliency-master/Classification/rebuttal_results/vit_b_cifar100_"

# distance = np.load(path + "distances.npy")
# orig_pred = np.load(path + "orig_pred.npy")
# ul_pred = np.load(path + "ul_pred.npy")
# gt = np.load(path + "gt.npy")

# path_2 = "/home/sazzad/Machine Unlearning/Unlearn-Saliency-master/Classification/rebuttal_results/vit_b_cifar100_forget_"
# nearest_classes = np.load(path_2 + "nearest_classes.npy")
# nearest_classes_distance = np.load(path_2 + "distances.npy")

# # print(np.unique(nearest_classes))

# # samples_misclassified = 1 - (ul_pred.ravel()==gt.ravel())

# samples_misclassified_after = 1 - (ul_pred.ravel()==gt.ravel())
# samples_misclassified_before = 1 - (orig_pred.ravel()==gt.ravel())
# samples_misclassified = np.array([1 if (x == 0 and y == 1) else 0 for x, y in zip(samples_misclassified_before, samples_misclassified_after)])

# # print(samples_misclassified)

# misclassified_ind = np.argwhere(samples_misclassified==1)
# labels_misclassified = gt[misclassified_ind]

# classes, count = np.unique(labels_misclassified, return_counts=True)

# # print(classes)

# args = np.argsort(count)[::-1]
# classes = classes[args]


# nearest_cls, nearest_cls_count = np.unique(nearest_classes, return_counts=True)

# # nearest_distance = [np.mean(nearest_classes_distance[nearest_classes==x]) for x in nearest_cls]
# # nearest_classes_distance = np.array(nearest_distance)

# neares_cls_ind = np.argsort(nearest_cls_count)[::-1]
# nearest_cls_dist = nearest_classes_distance[neares_cls_ind]
# nearest_cls = nearest_cls[neares_cls_ind]

# print(nearest_cls[:10])

# # print("\n\n")

# print(classes[:10])
# # print(count[:10])

# # from scipy import stats

# # res = stats.spearmanr(nearest_cls[:10], classes[:10])

# # print(res.statistic)

# # print(count[args] / np.sum(count))

# raise Exception

# same_cluster = [11, 35, 46, 98]

# within_cluster = [ 1 if x in same_cluster else 0 for x in labels_misclassified ]

# print(sum(within_cluster) / len(labels_misclassified))

# print(len(labels_misclassified))




        