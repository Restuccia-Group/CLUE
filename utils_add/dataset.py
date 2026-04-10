import os
import torch
from torchvision import datasets
from torchvision.transforms import v2 as transforms

import numpy as np

cur_dir = os.getcwd()
parent_dir = os.path.dirname(cur_dir)

class CIFAR100Coarse(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=True, use_coarse=False):
        super(CIFAR100Coarse, self).__init__(root, train, transform, target_transform, download)

        # update labels
        coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                                   3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                                   6, 11,  5, 10,  7,  6, 13, 15,  3, 15, 
                                   0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                                   5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                                   16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                                   10, 3,  2, 12, 12, 16, 12,  1,  9, 19, 
                                   2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                                  16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                                  18,  1,  2, 15,  6,  0, 17,  8, 14, 13])

        if use_coarse:
            self.targets = coarse_labels[self.targets]

        # update classes
        self.classes = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                        ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                        ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                        ['bottle', 'bowl', 'can', 'cup', 'plate'],
                        ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                        ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                        ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                        ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                        ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                        ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                        ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                        ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                        ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                        ['crab', 'lobster', 'snail', 'spider', 'worm'],
                        ['baby', 'boy', 'girl', 'man', 'woman'],
                        ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                        ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                        ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                        ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                        ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]

def get_dataset_from_code(code, data_folder_path=None, train_batch_size = 64, train_shuffle = True, get_val=True, val_split=0.2, val_from_test=False, 
                         val_batch_size=128, val_shuffle=False, test_batch_size = 128, test_shuffle=False, get_subset = False, 
                         subset_classes = None, use_coarse=False, root_dir=None, transforms_train=None, transforms_test=None):
    
    """root_dir is a must for the imagenet"""
    if data_folder_path is None:
        data_folder_path = os.path.join(parent_dir, f"data/{code}-data")
    if not get_val:
        if code=="mnist":
            train_loader, test_loader = get_mnist_dataloader(data_folder_path = data_folder_path, 
                                                             train_batch_size = train_batch_size, train_shuffle = train_shuffle, 
                                                             get_val=get_val, val_split=val_split, val_from_test=val_from_test, 
                                                             val_batch_size=val_batch_size, val_shuffle=val_shuffle, 
                                                             test_batch_size = test_batch_size, 
                                                             test_shuffle=test_shuffle, get_subset = get_subset, 
                                                             subset_classes = subset_classes, 
                                                             transforms_train=transforms_train, transforms_test=transforms_test
                                                             )
        elif code=="cifar10":
            train_loader, test_loader = get_cifar10_dataloader(data_folder_path = data_folder_path, 
                                                             train_batch_size = train_batch_size, train_shuffle = train_shuffle, 
                                                             get_val=get_val, val_split=val_split, val_from_test=val_from_test, 
                                                             val_batch_size=val_batch_size, val_shuffle=val_shuffle, 
                                                             test_batch_size = test_batch_size, 
                                                             test_shuffle=test_shuffle, get_subset = get_subset, 
                                                             subset_classes = subset_classes,
                                                             transforms_train=transforms_train, transforms_test=transforms_test
                                                             )
        elif code=="cifar100":
            train_loader, test_loader = get_cifar100_dataloader(data_folder_path = data_folder_path, 
                                                             train_batch_size = train_batch_size, train_shuffle = train_shuffle, 
                                                             get_val=get_val, val_split=val_split, val_from_test=val_from_test, 
                                                             val_batch_size=val_batch_size, val_shuffle=val_shuffle, 
                                                             test_batch_size = test_batch_size, 
                                                             test_shuffle=test_shuffle, get_subset = get_subset, 
                                                             subset_classes = subset_classes, use_coarse=use_coarse,
                                                             transforms_train=transforms_train, transforms_test=transforms_test
                                                             )
        elif code=="fmnist":
            train_loader, test_loader = get_fmnist_dataloader(data_folder_path = data_folder_path, 
                                                             train_batch_size = train_batch_size, train_shuffle = train_shuffle, 
                                                             get_val=get_val, val_split=val_split, val_from_test=val_from_test, 
                                                             val_batch_size=val_batch_size, val_shuffle=val_shuffle, 
                                                             test_batch_size = test_batch_size, 
                                                             test_shuffle=test_shuffle, get_subset = get_subset, 
                                                             subset_classes = subset_classes,
                                                             transforms_train=transforms_train, transforms_test=transforms_test
                                                             )
        elif code=="svhn":
            train_loader, test_loader = get_svhn_dataloader(data_folder_path = data_folder_path, 
                                                             train_batch_size = train_batch_size, train_shuffle = train_shuffle, 
                                                             get_val=get_val, val_split=val_split, val_from_test=val_from_test, 
                                                             val_batch_size=val_batch_size, val_shuffle=val_shuffle, 
                                                             test_batch_size = test_batch_size, 
                                                             test_shuffle=test_shuffle, get_subset = get_subset, 
                                                             subset_classes = subset_classes,
                                                             transforms_train=transforms_train, transforms_test=transforms_test
                                                             )
            
        elif code=="imagenet":
            train_loader, test_loader = get_imagenet_dataloader(root_dir=root_dir, distributed=False, batch_size=[train_batch_size, test_batch_size], 
                                                                workers=1, get_val=False, val_split=0.2, val_from_test=False, 
                                                                val_shuffle=False, test_shuffle=False, get_subset = False, subset_target_vals = None,
                                                                transforms_train=transforms_train, transforms_test=transforms_test
                                                                )
            
        elif code=="textures":
            train_loader, test_loader = get_textures_dataloader(data_folder_path = data_folder_path, 
                                                             train_batch_size = train_batch_size, train_shuffle = train_shuffle, 
                                                             get_val=get_val, val_split=val_split, val_from_test=val_from_test, 
                                                             val_batch_size=val_batch_size, val_shuffle=val_shuffle, 
                                                             test_batch_size = test_batch_size, 
                                                             test_shuffle=test_shuffle, get_subset = get_subset, 
                                                             subset_classes = subset_classes,
                                                             transforms_train=transforms_train, transforms_test=transforms_test
                                                             )
            
        elif code=="places":
            train_loader, test_loader = get_places_dataloader(data_folder_path = data_folder_path, 
                                                             train_batch_size = train_batch_size, train_shuffle = train_shuffle, 
                                                             get_val=get_val, val_split=val_split, val_from_test=val_from_test, 
                                                             val_batch_size=val_batch_size, val_shuffle=val_shuffle, 
                                                             test_batch_size = test_batch_size, 
                                                             test_shuffle=test_shuffle, get_subset = get_subset, 
                                                             subset_classes = subset_classes,
                                                             transforms_train=transforms_train, transforms_test=transforms_test
                                                             )
            
        elif code=="lsun":
            train_loader, test_loader = get_lsun_dataloader(data_folder_path = data_folder_path, 
                                                             train_batch_size = train_batch_size, train_shuffle = train_shuffle, 
                                                             get_val=get_val, val_split=val_split, val_from_test=val_from_test, 
                                                             val_batch_size=val_batch_size, val_shuffle=val_shuffle, 
                                                             test_batch_size = test_batch_size, 
                                                             test_shuffle=test_shuffle, get_subset = get_subset, 
                                                             subset_classes = subset_classes,
                                                             transforms_train=transforms_train, transforms_test=transforms_test
                                                             )
        elif code=="custom":
            train_loader, test_loader = get_custom_dataloader(data_folder_path = root_dir, 
                                                             train_batch_size = train_batch_size, train_shuffle = train_shuffle, 
                                                             get_val=get_val, val_split=val_split, val_from_test=val_from_test, 
                                                             val_batch_size=val_batch_size, val_shuffle=val_shuffle, 
                                                             test_batch_size = test_batch_size, 
                                                             test_shuffle=test_shuffle, get_subset = get_subset, 
                                                             subset_classes = subset_classes,
                                                             transforms_train=transforms_train, transforms_test=transforms_test
                                                             )
        else:
            raise ValueError(f"Unknown dataset name : {code}")
            
        return train_loader, test_loader
    else:
        if code=="mnist":
            train_loader, val_loader, test_loader = get_mnist_dataloader(data_folder_path = data_folder_path, 
                                                             train_batch_size = train_batch_size, train_shuffle = train_shuffle, 
                                                             get_val=get_val, val_split=val_split, val_from_test=val_from_test, 
                                                             val_batch_size=val_batch_size, val_shuffle=val_shuffle, 
                                                             test_batch_size = test_batch_size, 
                                                             test_shuffle=test_shuffle, get_subset = get_subset, 
                                                             subset_classes = subset_classes,
                                                             transforms_train=transforms_train, transforms_test=transforms_test
                                                             )
        elif code=="cifar10":
            train_loader, val_loader, test_loader = get_cifar10_dataloader(data_folder_path = data_folder_path, 
                                                             train_batch_size = train_batch_size, train_shuffle = train_shuffle, 
                                                             get_val=get_val, val_split=val_split, val_from_test=val_from_test, 
                                                             val_batch_size=val_batch_size, val_shuffle=val_shuffle, 
                                                             test_batch_size = test_batch_size, 
                                                             test_shuffle=test_shuffle, get_subset = get_subset, 
                                                             subset_classes = subset_classes,
                                                             transforms_train=transforms_train, transforms_test=transforms_test
                                                             )
        elif code=="cifar100":
            train_loader, val_loader, test_loader = get_cifar100_dataloader(data_folder_path = data_folder_path, 
                                                             train_batch_size = train_batch_size, train_shuffle = train_shuffle, 
                                                             get_val=get_val, val_split=val_split, val_from_test=val_from_test, 
                                                             val_batch_size=val_batch_size, val_shuffle=val_shuffle, 
                                                             test_batch_size = test_batch_size, 
                                                             test_shuffle=test_shuffle, get_subset = get_subset, 
                                                             subset_classes = subset_classes, use_coarse=use_coarse,
                                                             transforms_train=transforms_train, transforms_test=transforms_test
                                                             )
        elif code=="fmnist":
            train_loader, val_loader, test_loader = get_fmnist_dataloader(data_folder_path = data_folder_path, 
                                                             train_batch_size = train_batch_size, train_shuffle = train_shuffle, 
                                                             get_val=get_val, val_split=val_split, val_from_test=val_from_test, 
                                                             val_batch_size=val_batch_size, val_shuffle=val_shuffle, 
                                                             test_batch_size = test_batch_size, 
                                                             test_shuffle=test_shuffle, get_subset = get_subset, 
                                                             subset_classes = subset_classes,
                                                             transforms_train=transforms_train, transforms_test=transforms_test
                                                             )
        elif code=="svhn":
            train_loader, val_loader, test_loader = get_svhn_dataloader(data_folder_path = data_folder_path, 
                                                             train_batch_size = train_batch_size, train_shuffle = train_shuffle, 
                                                             get_val=get_val, val_split=val_split, val_from_test=val_from_test, 
                                                             val_batch_size=val_batch_size, val_shuffle=val_shuffle, 
                                                             test_batch_size = test_batch_size, 
                                                             test_shuffle=test_shuffle, get_subset = get_subset, 
                                                             subset_classes = subset_classes,
                                                             transforms_train=transforms_train, transforms_test=transforms_test
                                                             )
            
        elif code=="imagenet":
            train_loader, val_loader, test_loader = get_imagenet_dataloader(root_dir=root_dir, distributed=False, batch_size=[train_batch_size, test_batch_size], 
                                                                workers=1, get_val=False, val_split=0.2, val_from_test=False, 
                                                                val_shuffle=False, test_shuffle=False, get_subset = False, subset_target_vals = None,
                                                                transforms_train=transforms_train, transforms_test=transforms_test
                                                                )
            
        elif code=="textures":
            train_loader, val_loader, test_loader = get_textures_dataloader(data_folder_path = data_folder_path, 
                                                             train_batch_size = train_batch_size, train_shuffle = train_shuffle, 
                                                             get_val=get_val, val_split=val_split, val_from_test=val_from_test, 
                                                             val_batch_size=val_batch_size, val_shuffle=val_shuffle, 
                                                             test_batch_size = test_batch_size, 
                                                             test_shuffle=test_shuffle, get_subset = get_subset, 
                                                             subset_classes = subset_classes,
                                                             transforms_train=transforms_train, transforms_test=transforms_test
                                                             )
            
        elif code=="places":
            train_loader, val_loader, test_loader = get_places_dataloader(data_folder_path = data_folder_path, 
                                                             train_batch_size = train_batch_size, train_shuffle = train_shuffle, 
                                                             get_val=get_val, val_split=val_split, val_from_test=val_from_test, 
                                                             val_batch_size=val_batch_size, val_shuffle=val_shuffle, 
                                                             test_batch_size = test_batch_size, 
                                                             test_shuffle=test_shuffle, get_subset = get_subset, 
                                                             subset_classes = subset_classes,
                                                             transforms_train=transforms_train, transforms_test=transforms_test
                                                             )
            
        elif code=="lsun":
            train_loader, val_loader, test_loader = get_lsun_dataloader(data_folder_path = data_folder_path, 
                                                             train_batch_size = train_batch_size, train_shuffle = train_shuffle, 
                                                             get_val=get_val, val_split=val_split, val_from_test=val_from_test, 
                                                             val_batch_size=val_batch_size, val_shuffle=val_shuffle, 
                                                             test_batch_size = test_batch_size, 
                                                             test_shuffle=test_shuffle, get_subset = get_subset, 
                                                             subset_classes = subset_classes,
                                                             transforms_train=transforms_train, transforms_test=transforms_test
                                                             )
        elif code=="custom":
            train_loader, val_loader, test_loader = get_custom_dataloader(data_folder_path = root_dir, 
                                                             train_batch_size = train_batch_size, train_shuffle = train_shuffle, 
                                                             get_val=get_val, val_split=val_split, val_from_test=val_from_test, 
                                                             val_batch_size=val_batch_size, val_shuffle=val_shuffle, 
                                                             test_batch_size = test_batch_size, 
                                                             test_shuffle=test_shuffle, get_subset = get_subset, 
                                                             subset_classes = subset_classes,
                                                             transforms_train=transforms_train, transforms_test=transforms_test
                                                             )
            
        else:
            raise ValueError(f"Unknown dataset name : {code}")
            
        return train_loader, val_loader, test_loader

def get_dataloader(train_set, test_set, train_batch_size, train_shuffle, get_val, val_split, val_from_test, 
                   val_batch_size, val_shuffle, test_batch_size, test_shuffle, get_subset, subset_classes):
    
    kwargs = {'num_workers': 8, 'pin_memory': True}
    
    if not get_val:
        
        if get_subset:
            train_indices = [i for i in range(len(train_set)) if train_set[i][1] in subset_classes]
            test_indices = [i for i in range(len(test_set)) if test_set[i][1] in subset_classes]
            
            train_set = torch.utils.data.Subset(train_set, train_indices)
            test_set = torch.utils.data.Subset(test_set, test_indices)

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=train_batch_size, shuffle=train_shuffle, **kwargs,
        )

        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=test_batch_size, shuffle=test_shuffle, **kwargs,
        )
        
        return train_loader, test_loader

    if val_from_test is True:
        if isinstance(val_split, float):
            val_split = int(val_split * len(test_set))
        test_set, val_set = torch.utils.data.random_split(
            test_set, [len(test_set) - val_split, val_split]
        )
    else:
        if isinstance(val_split, float):
            val_split = int(val_split * len(test_set))
        train_set, val_set = torch.utils.data.random_split(
            train_set, [len(train_set) - val_split, val_split]
        )

    if get_subset:
        train_indices = [i for i in range(len(train_set)) if train_set[i][1] in subset_classes]
        val_indices = [i for i in range(len(val_set)) if val_set[i][1] in subset_classes]
        test_indices = [i for i in range(len(test_set)) if test_set[i][1] in subset_classes]
        
        train_set = torch.utils.data.Subset(train_set, train_indices)
        val_set = torch.utils.data.Subset(val_set, val_indices)
        test_set = torch.utils.data.Subset(test_set, test_indices)
         
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=train_batch_size, shuffle=train_shuffle, **kwargs,
    )

    validation_loader = torch.utils.data.DataLoader(
        val_set, batch_size=val_batch_size, shuffle=val_shuffle, **kwargs,
    )

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=test_batch_size, shuffle=test_shuffle, **kwargs,
    )

    return train_loader, validation_loader, test_loader


#################################################################
#                     MNIST DataLoader                          #
#################################################################


def get_mnist_dataloader(data_folder_path, train_batch_size, train_shuffle, get_val, val_split, val_from_test, 
                         val_batch_size, val_shuffle, test_batch_size, test_shuffle, get_subset, 
                         subset_classes, transforms_train=None, transforms_test=None):
    
    """mnist datalaoder
    Args:
        data_folder_path(path)  : This is the path from where the data will be fetched
        train_batch_size(int)   : Size of training batches
        train_shuffle(bool)     : if True then training sample will be shuffled otherwise not
        get_val(bool)           : if true then validation set wii=ll be generated
        val_split(int or float) : If an integer, then val_split % samples will be separated for validation
        val_from_test(bool)     : If True, then validation samples will be collected from test set otherwise from train set
        val_batch_size(int)     : Size of validation batches
        val_shuffle(bool)       : If true the validation samples will be shuffled
        testing_batch_size(int) : Size of testing  batches
        test_shuffle(bool)      : If true the test samples will be shuffled
        get_subset(bool)        : If True, only a given subset of the dataset will be loaded
        subset_classes(List)    : The classes which are to be included in dataset if get_subset is True 
    """
    # normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2675, 0.2565, 0.2761])
    
    if transforms_train is None:
        transforms_train = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize(32),
            transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
            #transforms.Normalize((0.1307,), (0.3081,))
            ])
        
    if transforms_test is None:
        transforms_test=transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize(32),
            transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
            #transforms.Normalize((0.1307,), (0.3081,))
            ])

    train_set = datasets.MNIST(data_folder_path, train=True,  download=True, 
        transform=transforms_train
        )

    test_set  = datasets.MNIST(data_folder_path, train=False, download=True, 
        transform=transforms_test
        )
    
    dataloaders = get_dataloader(train_set, test_set, train_batch_size = train_batch_size, train_shuffle = train_shuffle, 
                                 get_val=get_val, val_split=val_split, val_from_test=val_from_test, 
                                 val_batch_size=val_batch_size, val_shuffle=val_shuffle, test_batch_size = test_batch_size,
                                 test_shuffle=test_shuffle, get_subset = get_subset, subset_classes = subset_classes)
    
    return dataloaders
   
    
    
    
#################################################################
#                  CIFAR 10 DataLoader                         #
#################################################################

def get_cifar10_dataloader(data_folder_path, train_batch_size, train_shuffle, get_val, val_split, val_from_test, 
                         val_batch_size, val_shuffle, test_batch_size, test_shuffle, get_subset, 
                         subset_classes, transforms_train=None, transforms_test=None):
    """cifar10 datalaoder
    Args:
        data_folder_path(path)  : This is the path from where the data will be fetched
        train_batch_size(int)   : Size of training batches
        train_shuffle(bool)     : if True then training sample will be shuffled otherwise not
        get_val(bool)           : if true then validation set wii=ll be generated
        val_split(int or float) : If an integer, then val_split % samples will be separated for validation
        val_from_test(bool)     : If True, then validation samples will be collected from test set otherwise from train set
        val_batch_size(int)     : Size of validation batches
        val_shuffle(bool)       : If true the validation samples will be shuffled
        testing_batch_size(int) : Size of testing  batches
        test_shuffle(bool)      : If true the test samples will be shuffled
        get_subset(bool)        : If True, only a given subset of the dataset will be loaded
        subset_classes(List)    : The classes which are to be included in dataset if get_subset is True 
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    
    if transforms_train is None:
        transforms_train = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize(32),
            transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
            #transforms.Normalize((0.1307,), (0.3081,))
            ])
        
    if transforms_test is None:
        transforms_test=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]
        )

    train_set = datasets.CIFAR10(
        data_folder_path,
        train=True,
        download=True,
        transform=transforms_test
    )

    test_set = datasets.CIFAR10(
        data_folder_path,
        train=False,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
    )
    
    dataloaders = get_dataloader(train_set, test_set, train_batch_size = train_batch_size, train_shuffle = train_shuffle, 
                                 get_val=get_val, val_split=val_split, val_from_test=val_from_test, 
                                 val_batch_size=val_batch_size, val_shuffle=val_shuffle, test_batch_size = test_batch_size,
                                 test_shuffle=test_shuffle, get_subset = get_subset, subset_classes = subset_classes)
    
    return dataloaders


#################################################################
#                  CIFAR 100 DataLoader                         #
#################################################################

def get_cifar100_dataloader(data_folder_path, train_batch_size, train_shuffle, get_val, val_split, val_from_test, 
                         val_batch_size, val_shuffle, test_batch_size, test_shuffle, get_subset, 
                         subset_classes, use_coarse, transforms_train=None, transforms_test=None):
    """cifar100 datalaoder
    Args:
        data_folder_path(path)  : This is the path from where the data will be fetched
        train_batch_size(int)   : Size of training batches
        train_shuffle(bool)     : if True then training sample will be shuffled otherwise not
        get_val(bool)           : if true then validation set wii=ll be generated
        val_split(int or float) : If an integer, then val_split % samples will be separated for validation
        val_from_test(bool)     : If True, then validation samples will be collected from test set otherwise from train set
        val_batch_size(int)     : Size of validation batches
        val_shuffle(bool)       : If true the validation samples will be shuffled
        testing_batch_size(int) : Size of testing  batches
        test_shuffle(bool)      : If true the test samples will be shuffled
        get_subset(bool)        : If True, only a given subset of the dataset will be loaded
        subset_classes(List)    : The classes which are to be included in dataset if get_subset is True 
        use_coarse(bool) : If True, the coarse class labels of CIFAR100 will be used 
    """
    normalize = transforms.Normalize(
        mean=[0.5071, 0.4865, 0.4409], std=[0.2675, 0.2565, 0.2761]
        )
    
    if transforms_train is None:
        transforms_train = transforms.Compose(
           [
             transforms.RandomHorizontalFlip(),
             transforms.RandomCrop(32, 4),
             transforms.ToTensor(),
             normalize
           ]
      )
        
    if transforms_test is None:
        transforms_test=transforms.Compose(
           [
             transforms.ToTensor(),
             normalize
           ]
      )

    train_set = CIFAR100Coarse(data_folder_path, train=True, download=True,\
                                   transform=transforms_train, use_coarse=use_coarse
                                )

    test_set = CIFAR100Coarse(
               data_folder_path, train=False, download=True, \
                                              transform=transforms_test, use_coarse=use_coarse)
    
    dataloaders = get_dataloader(train_set, test_set, train_batch_size = train_batch_size, train_shuffle = train_shuffle, 
                                 get_val=get_val, val_split=val_split, val_from_test=val_from_test, 
                                 val_batch_size=val_batch_size, val_shuffle=val_shuffle, test_batch_size = test_batch_size,
                                 test_shuffle=test_shuffle, get_subset = get_subset, subset_classes = subset_classes)
    
    return dataloaders

#################################################################
#                  Fashion Mnist DataLoader                     #
#################################################################

def get_fmnist_dataloader(data_folder_path, train_batch_size, train_shuffle, get_val, val_split, val_from_test, 
                         val_batch_size, val_shuffle, test_batch_size, test_shuffle, get_subset, 
                         subset_classes, transforms_train=None, transforms_test=None):
    """Fashion mnist datalaoder
    Args:
        data_folder_path(path)  : This is the path from where the data will be fetched
        train_batch_size(int)   : Size of training batches
        train_shuffle(bool)     : if True then training sample will be shuffled otherwise not
        get_val(bool)           : if true then validation set wii=ll be generated
        val_split(int or float) : If an integer, then val_split % samples will be separated for validation
        val_from_test(bool)     : If True, then validation samples will be collected from test set otherwise from train set
        val_batch_size(int)     : Size of validation batches
        val_shuffle(bool)       : If true the validation samples will be shuffled
        testing_batch_size(int) : Size of testing  batches
        test_shuffle(bool)      : If true the test samples will be shuffled
        get_subset(bool)        : If True, only a given subset of the dataset will be loaded
        subset_classes(List)    : The classes which are to be included in dataset if get_subset is True 
    """
    
    if transforms_train is None:
        transforms_train = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(32),
                                        transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                                       #transforms.Normalize((0.2860,), (0.3530,)),
                                     ])
        
    if transforms_test is None:
        transforms_test=transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(32),
                                        transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                                       #transforms.Normalize((0.2860,), (0.3530,)),
                                     ])
    # Download and load the training data
    train_set = datasets.FashionMNIST(data_folder_path, download=True, train=True, transform=transforms_train)

    test_set = datasets.FashionMNIST(data_folder_path, download=True, train=False, transform=transforms_test)
    
    dataloaders = get_dataloader(train_set, test_set, train_batch_size = train_batch_size, train_shuffle = train_shuffle, 
                                 get_val=get_val, val_split=val_split, val_from_test=val_from_test, 
                                 val_batch_size=val_batch_size, val_shuffle=val_shuffle, test_batch_size = test_batch_size,
                                 test_shuffle=test_shuffle, get_subset = get_subset, subset_classes = subset_classes)
    
    return dataloaders
        
        
#################################################################
#                       SVHN DataLoader                         #
#################################################################   
        
        
def get_svhn_dataloader(data_folder_path, train_batch_size, train_shuffle, get_val, val_split, val_from_test, 
                         val_batch_size, val_shuffle, test_batch_size, test_shuffle, get_subset, 
                         subset_classes, transforms_train=None, transforms_test=None):
    
    if transforms_train is None:
        transforms_train = transforms.Compose([transforms.ToTensor()])
        
    if transforms_test is None:
        transforms_test=transforms.Compose([transforms.ToTensor()])
    
    train_set = datasets.SVHN(root=data_folder_path, download=True, split="train", transform=transforms_train)
    
    test_set = datasets.SVHN(root=data_folder_path, download=True, split="test", transform=transforms_test)
    
    dataloaders = get_dataloader(train_set, test_set, train_batch_size = train_batch_size, train_shuffle = train_shuffle, 
                                 get_val=get_val, val_split=val_split, val_from_test=val_from_test, 
                                 val_batch_size=val_batch_size, val_shuffle=val_shuffle, test_batch_size = test_batch_size,
                                 test_shuffle=test_shuffle, get_subset = get_subset, subset_classes = subset_classes)
    
    return dataloaders

#################################################################
#                     Textures DataLoader                       #
################################################################# 
def get_textures_dataloader(data_folder_path, train_batch_size, train_shuffle, get_val, val_split, val_from_test, 
                         val_batch_size, val_shuffle, test_batch_size, test_shuffle, get_subset, 
                         subset_classes, transforms_train=None, transforms_test=None):
    print("In Here")
    if transforms_train is None:
        transforms_train = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        
    if transforms_test is None:
        transforms_test = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        
    train_set = datasets.DTD(root=data_folder_path, download=True, split="train", transform=transforms_train)
    
    test_set = datasets.DTD(root=data_folder_path, download=True, split="test", transform=transforms_test)
    
    dataloaders = get_dataloader(train_set, test_set, train_batch_size = train_batch_size, train_shuffle = train_shuffle, 
                                 get_val=get_val, val_split=val_split, val_from_test=val_from_test, 
                                 val_batch_size=val_batch_size, val_shuffle=val_shuffle, test_batch_size = test_batch_size,
                                 test_shuffle=test_shuffle, get_subset = get_subset, subset_classes = subset_classes)
    return dataloaders


#################################################################
#                     Places365 DataLoader                      #
################################################################# 
def get_places_dataloader(data_folder_path, train_batch_size, train_shuffle, get_val, val_split, val_from_test, 
                         val_batch_size, val_shuffle, test_batch_size, test_shuffle, get_subset, 
                         subset_classes, transforms_train=None, transforms_test=None):
    
    if transforms_train is None:
        transforms_train = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
        
    if transforms_test is None:
        transforms_test = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
        
    train_set = datasets.Places365(root=data_folder_path, download=True, split="train-standard", transform=transforms_train)
    
    test_set = datasets.Places365(root=data_folder_path, download=True, split="val", transform=transforms_test)
    
    dataloaders = get_dataloader(train_set, test_set, train_batch_size = train_batch_size, train_shuffle = train_shuffle, 
                                 get_val=get_val, val_split=val_split, val_from_test=val_from_test, 
                                 val_batch_size=val_batch_size, val_shuffle=val_shuffle, test_batch_size = test_batch_size,
                                 test_shuffle=test_shuffle, get_subset = get_subset, subset_classes = subset_classes)
    return dataloaders

#################################################################
#                         LSUN DataLoader                       #
################################################################# 
def get_lsun_dataloader(data_folder_path, train_batch_size, train_shuffle, get_val, val_split, val_from_test, 
                         val_batch_size, val_shuffle, test_batch_size, test_shuffle, get_subset, 
                         subset_classes, transforms_train=None, transforms_test=None):
    """Need lmdb package for this dataset """
    
    if transforms_train is None:
        transforms_train = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
        
    if transforms_test is None:
        transforms_test = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
        
    train_set = datasets.LSUN(root=data_folder_path, split="train", transform=transforms_train)
    
    test_set = datasets.LSUN(root=data_folder_path, split="test", transform=transforms_test)
    
    dataloaders = get_dataloader(train_set, test_set, train_batch_size = train_batch_size, train_shuffle = train_shuffle, 
                                 get_val=get_val, val_split=val_split, val_from_test=val_from_test, 
                                 val_batch_size=val_batch_size, val_shuffle=val_shuffle, test_batch_size = test_batch_size,
                                 test_shuffle=test_shuffle, get_subset = get_subset, subset_classes = subset_classes)
    return dataloaders

#################################################################
#                       Imagenet DataLoader                     #
#################################################################   
        
def custom_imagenet_dataset(root_dir, distributed=False, transforms_train=None, transforms_test=None):
    
    train_dir = os.path.join(root_dir, 'train')
    test_dir = os.path.join(root_dir, 'val')
    
    # Define data transformations
    
    if transforms_train is None:
        transforms_train = transforms.Compose([
            transforms.Resize(256),         # Resize the input image to 256x256 pixels
            transforms.CenterCrop(224),     # Crop the center 224x224 pixels from the resized image
            transforms.ToTensor(),          # Convert the image to a PyTorch tensor
            transforms.Normalize(           # Normalize the image's pixel values
                mean=[0.485, 0.456, 0.406],  # Mean values for the three color channels
                std=[0.229, 0.224, 0.225]    # Standard deviation values for the three color channels
            )
        ])
        
    if transforms_test is None:
        transforms_test = transforms.Compose([
            transforms.Resize(256),         # Resize the input image to 256x256 pixels
            transforms.CenterCrop(224),     # Crop the center 224x224 pixels from the resized image
            transforms.ToTensor(),          # Convert the image to a PyTorch tensor
            transforms.Normalize(           # Normalize the image's pixel values
                mean=[0.485, 0.456, 0.406],  # Mean values for the three color channels
                std=[0.229, 0.224, 0.225]    # Standard deviation values for the three color channels
            )
        ])
    
    train_dataset = datasets.ImageFolder(train_dir, transforms_train)
    test_dataset = datasets.ImageFolder(test_dir, transforms_test)
    
    return train_dataset, test_dataset

def get_imagenet_dataloader(root_dir, distributed=False, batch_size=16, workers=1, 
                            get_val=False, val_split=0.2, val_from_test=False, val_shuffle=False, test_shuffle=False,
                            get_subset = False, subset_target_vals = None, transforms_train=None, transforms_test=None):
    
    if isinstance(batch_size, list):
        if len(batch_size)==2:
            train_batch_size = batch_size[0]
            test_batch_size = batch_size[1]
        elif len(batch_size)==3:
            train_batch_size = batch_size[0]
            val_batch_size = batch_size[1]
            test_batch_size = batch_size[2]
    else:
        train_batch_size = batch_size
        val_batch_size = batch_size
        test_batch_size = batch_size
    
    train_dataset, test_dataset = custom_imagenet_dataset(root_dir, distributed=distributed, 
                                                          transforms_train=transforms_train, transforms_test=transforms_test)
    if get_subset:
        train_indices = None
        test_indices = None
        for i in subset_target_vals:
            if train_indices is not None:
                train_indice = torch.tensor(train_dataset.targets) == i
                train_indices = train_indices | train_indice
            else:
                train_indice = torch.tensor(train_dataset.targets)==i
                train_indices = train_indice
            if test_indices is not None:
                test_indice = torch.tensor(test_dataset.targets) == i
                test_indices = test_indice | test_indice
            else:
                test_indice = torch.tensor(test_dataset.targets)==i
                test_indices = test_indice
        train_indices = train_indices.nonzero().reshape(-1)
        test_indices = test_indices.nonzero().reshape(-1)
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
        
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler((train_dataset))
    else:
        train_sampler = None
        
    if not get_val:
        dataloaders = get_dataloader(train_dataset, test_dataset, train_batch_size = train_batch_size, 
                                     train_shuffle = (train_sampler is None), get_val=get_val, 
                                     val_split=val_split, val_from_test=val_from_test, 
                                     val_batch_size = None, val_shuffle=val_shuffle, test_batch_size = test_batch_size,
                                     test_shuffle=test_shuffle, get_subset = get_subset, subset_classes = subset_target_vals)
    else:
        dataloaders = get_dataloader(train_dataset, test_dataset, train_batch_size = train_batch_size, 
                                     train_shuffle = (train_sampler is None), get_val=get_val, 
                                     val_split=val_split, val_from_test=val_from_test, 
                                     val_batch_size = val_batch_size, val_shuffle=val_shuffle, test_batch_size = test_batch_size,
                                     test_shuffle=test_shuffle, get_subset = get_subset, subset_classes = subset_target_vals)
    
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=train_batch_size, shuffle=(train_sampler is None),
    #     num_workers=workers, pin_memory=True, sampler=train_sampler)
    
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size)
    
    
    return dataloaders
        
#################################################################
#                       Custom DataLoader                       #
#################################################################  

def custom_dataset(root_dir, distributed=False, transforms_train=None, transforms_test=None):
    
    train_dir = os.path.join(root_dir, 'train')
    test_dir = os.path.join(root_dir, 'test')
        
    train_dataset = datasets.ImageFolder(train_dir, transforms_train)
    test_dataset = datasets.ImageFolder(test_dir, transforms_test)
    
    return train_dataset, test_dataset
        
        
def get_custom_dataloader(data_folder_path, train_batch_size, train_shuffle, get_val, val_split, val_from_test, 
                         val_batch_size, val_shuffle, test_batch_size, test_shuffle, get_subset, 
                         subset_classes, transforms_train=None, transforms_test=None):
    """Need lmdb package for this dataset """
    
    if transforms_train is None:
        transforms_train = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
        
    if transforms_test is None:
        transforms_test = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
        
    train_set, test_set = custom_dataset(data_folder_path, transforms_train=transforms_train, transforms_test=transforms_test)
    
    dataloaders = get_dataloader(train_set, test_set, train_batch_size = train_batch_size, train_shuffle = train_shuffle, 
                                 get_val=get_val, val_split=val_split, val_from_test=val_from_test, 
                                 val_batch_size=val_batch_size, val_shuffle=val_shuffle, test_batch_size = test_batch_size,
                                 test_shuffle=test_shuffle, get_subset = get_subset, subset_classes = subset_classes)
    return dataloaders 
        
        
        
        
        
        
        