
import os
import pandas as pd
import torch
from PIL import Image
import torch.nn.parallel
import torch.optim
# import torch.utils.data
import torchvision.transforms as transforms
# from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader, DistributedSampler
# from imagecorruptions import get_corruption_names
from torch.utils.data.sampler import Sampler
import numpy as np
from numpy.random import dirichlet
from conf import cfg
# gen = torch.Generator()
# gen.manual_seed(0)  

# args={}
# args['data_dir'] = 'Augmented'
# args['check_pt_noise_path'] = 'saved_model/noise_embedding_ckpt_2.pt'
# args['log_path_noise'] = 'LOGS/'
# args['backbone_model_dir'] = 'saved_model/saved_model_new.pt'
# args['tr_mode'] = 'testing'
# args['n_epochs'] = 500
# args['batch_size'] = 64
# args['temperature'] = 0.1
# args['weight_decay'] = .001
# args['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# args['learning_rate'] = 0.01  
# args['lambda_embed'] = 0.005 

# cifar10 =  CIFAR10(root = '/home/rifat/Documents/Python Scripts/CIFAR_10/data/', download=True)
class CorrelatedSampler(Sampler):
    def __init__(self, data_list: pd.DataFrame, batch_size=64, gamma=10, slots= None):
        
        self.data_list = data_list
        self.domains = list(data_list['noise'].unique())
        self.classes = list(data_list['target'].unique())
        self.num_classes = len(self.classes)
        self.gamma = gamma
        if slots is not None:
            self.num_slots = slots
        else:
            self.num_slots = self.num_classes if self.num_classes <=100 else 100
    
    def __len__(self):
        return len(self.data_list)
    
    def __iter__(self):
        
        final_indices = []
        for domain in self.domains:
            indices = self.data_list[self.data_list['noise'] == domain].index.to_numpy()
            labels = np.array(self.data_list.iloc[indices]['target'])
            
            
            #class_indices = np.array([self.data_list.iloc[indices]['target']== _class for _class in self.classes])
            class_indices = [np.argwhere(labels==y).flatten() for y in range(self.num_classes)] #list of n_class arrays with class indices
            slot_indices = [[] for _ in range (self.num_slots)]
            
            label_distribution = dirichlet([self.gamma]*self.num_slots, self.num_classes)
            
            for c_ids, partition in zip(class_indices, label_distribution):
                for s, ids in enumerate(np.split(c_ids, (np.cumsum(partition)[:-1]*len(c_ids)).astype(int))):
                    slot_indices[s].extend(ids)
                    # print(len(slot_indices[s]))
                # raise Exception("Stop - line 69")
            
            for s_ids in slot_indices:
                permutation = np.random.permutation(range(len(s_ids)))
                ids = []
                
                for i in permutation:
                    ids.append(s_ids[i])
                    # print(s_ids[i])
                
                
                final_indices.extend(indices[ids])
                # print(type(indices[ids]))
                # raise Exception("Stop - line 76")
                
        
        return iter(final_indices)
        


cifar10_classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

seen_corruption = ['brightness','contrast','defocus_blur','elastic_transform',
                   'fog','frost','gaussian_noise',
                   'glass_blur','impulse_noise','jpeg_compression','motion_blur','normal','pixelate',
                   'shot_noise','snow','zoom_blur']

unseen_corruption = ['gaussian_blur','saturate','spatter','speckle_noise']
#aug_set = ['brightness', 'contrast','saturation', 'hue', 'gamma','Gaussian_Blur', 'Gaussian_Noise']
# data_list = create_df_cifar10(filepath=args['data_dir'],noise=unseen_corruption,mode='test')
# sampler=CorrelatedSampler(data_list=data_list)

# domains = list(data_list['noise'].unique())
# classes = list(data_list['target'].unique())
# num_classes = len(classes)
# gamma = 0.1

# num_slots = num_classes if num_classes <=100 else 100

# final_indices = []
# for domain in domains:
#      indices = data_list[data_list['noise'] == domain].index.to_numpy()
#      labels = np.array(data_list.iloc[indices]['target'])
     
#      #class_indices = np.array([data_list.iloc[indices]['target']== _class for _class in classes])
#      class_indices = [np.argwhere(labels==y).flatten() for y in range(num_classes)]
#      slot_indices = [[] for _ in range (num_slots)]
     
#      label_distribution = dirichlet([gamma]*num_slots, num_classes)
     
#      for c_ids, partition in zip(class_indices, label_distribution):
#          for s, ids in enumerate(np.split(c_ids, (np.cumsum(partition)[:-1]*len(c_ids)).astype(int))):
#              slot_indices[s].append(ids)
#      for s_ids in slot_indices:
#          permutation = np.random.permutation(range(len(s_ids)))
#          ids = []
#          for i in permutation:
#              ids.extend(s_ids[i])
#          final_indices.extend(indices[ids])

# def get_loader(args, mode='train', tr_mode='contrastive'):
#     #change hard coded data_dir and noise
#     dataset = Dataset_Corruption_FingerPrint(data_dir='Common_Noise/',
#                                 noise=None,
#                                 transform=get_transform(args, mode=mode, tr_mode=tr_mode),
#                                 mode=mode)
#     shuffle=True if mode=='train' else False
    
#     data_loader = DataLoader(dataset=dataset, 
#                              batch_size = args['batch_size'], 
#                              shuffle=shuffle,
#                              num_workers=4)
#     return data_loader

def get_loader(cfg, mode= 'train', corelated=False, batch_size = 64, gamma=0.1, noise=None):

    data_list = create_df_cifar10(cfg, filepath=cfg.PATH.DATA_PATH,
                                  noise=noise, mode=mode)
    assert len(data_list) !=0 , "No data found in the data list"
    dataset = Dataset_Corrupted( 
        #data_dir = args['data_dir'],
        data_list=data_list,
        noise = noise,
        transform = get_transform(tr_mode= cfg.TR.MODE) if mode=='train' else None,
        mode = mode
        )
    
    # dataset = Dataset_Corrupted_Full( 
    #      data_dir = args['data_dir'],
    #      noise = noise,
    #      transform = get_transform(tr_mode= args['tr_mode']),
    #      mode = mode
    #      )
    shuffle = True if mode == 'train' else False
    #shuffle = True
    data_loader = DataLoader(dataset=dataset, 
                              batch_size = batch_size, 
                              shuffle=shuffle,
                              num_workers=cfg.BASE.NUM_WORKERS)
    # data_loader = DataLoader(dataset=dataset, 
    #                           batch_size = args['batch_size'], 
    #                           shuffle=False,
    #                           sampler= DistributedSampler(dataset),
    #                           num_workers=8)
    if corelated and mode =='test':
        data_loader = DataLoader(dataset=dataset, 
                              batch_size = cfg.TST.BATCH_SIZE, 
                              shuffle=shuffle,
                              #sampler=CorrelatedSampler(data_list=data_list,gamma=gamma), #gamma value
                              num_workers=cfg.BASE.NUM_WORKERS)
        
    return data_loader


def get_transform(tr_mode = 'training'):
    assert tr_mode in ['training','testing','noise_contrast'], "Wrong tr_mode inserted"
    transform_train = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
        ]
    transform_train = transforms.Compose(transform_train)
    
    # transform_noise_contrast = transforms.RandomApply(
    #     torch.nn.ModuleList(
    #         [transform.RandomHorizontalFlip(),
    #          transforms.RandomVerticalFlip(),
    #          transforms.RandomRotation(degrees=(0,180))
    #         ]), p=0.3
    #     )
    transform_noise_contrast = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=(0,180))]
    
    transform_noise_contrast = transforms.Compose(transform_noise_contrast)
    
    if tr_mode == 'training':
        return transform_train
    elif tr_mode == 'noise_contrast':
        return transform_noise_contrast
    else:
        return None
        

# class create_dataset:
    
#     def __init__(self, filepath, noise = None, mode= 'train'):
#         filelist = []
#         for root, dirs, files in os.walk(filepath):
#             for file in files:
#                 filelist.append(os.path.normpath(os.path.join(root,file)).split(os.path.sep))
#         columns = ['root','noise','mode','class','sample']
#         df = pd.DataFrame(filelist, columns=columns)
            
#         if noise is not None:
#             noise = [noise] if not isinstance(noise, list) else noise
#             df = df[df['noise'].isin(noise)]
                
#         df ['path'] = (df['root'].astype(str) + '/' + 
#                             df['noise'].astype(str) + '/' +
#                             df['mode'].astype(str) + '/' +
#                             df['class'].astype(str) + '/' +
#                             df['sample'].astype(str)  )
            
#         df['target'] = df['class'].map(lambda x: cifar10_classes.index(x))
#         # le = LabelEncoder()
#         # le.fit(df['noise'].unique())
#         # df['noise'] = le.transform(df['noise'])
#         if 'common' in filepath:
#             df['noise'] = df['noise'].map(lambda x: seen_corruption.index(x))
#         else:
#             df['noise'] = df['noise'].map(lambda x: unseen_corruption.index(x))
            
#         self.df = df[df['mode']==mode]
           
#     def get_data(self):
#         self.df['data'] = self.df['path'].map(self.read_img)
    
#     @property 
#     def data(self):
#         return self.df
    
#     def read_img(self, path):
#         to_tensor = []
#         ## Resize the tensor here if needed
#         to_tensor += [transforms.ToTensor()]
#         to_tensor += [transforms.Normalize(
#             mean = (0.4914, 0.4822, 0.4465),
#             std = (0.2023, 0.1994, 0.2010)) # (0.2470, 0.2435, 0.2616)
#             ]
#         to_tensor = transforms.Compose(to_tensor) 
#         img = Image.open(path)
#         img = to_tensor(img)
#         return img
    
def create_df_cifar10(cfg,filepath, noise=None, mode='train'):
    filelist = []
    assert os.path.isdir(filepath), 'Invalid path to dataset is inserted'
    for root, _ , files in os.walk(filepath):
        for file in files:
            filelist.append(os.path.normpath(os.path.join(root,file)).split(os.path.sep))
    columns = ['root','severity','noise','mode','class','sample']
    df = pd.DataFrame(filelist, columns=columns)
        
    if noise is not None:
        noise = [noise] if not isinstance(noise, list) else noise
        df = df[df['noise'].isin(noise)]
            
    df ['path'] = (df['root'].astype(str) + '/' + 
                        df['severity'].astype(str) + '/' +
                        df['noise'].astype(str) + '/' +
                        df['mode'].astype(str) + '/' +
                        df['class'].astype(str) + '/' +
                        df['sample'].astype(str)  )
    
    df = df[df['mode']==mode]  
    
    df['target'] = df['class'].map(lambda x: cifar10_classes.index(x))
    df['noise'] = df['noise'].map(lambda x: cfg.DATA.CORRUPTION_SET.index(x)) # change here
    df = df.reset_index(drop=True)   
    return df

class Dataset_Corrupted(Dataset):
    def __init__(self, data_list,noise='normal',transform=None,mode='train') -> None:
        #self.data_dir = data_dir
        #self.annotations = create_df_cifar10(filepath=data_dir,noise=noise,mode=mode)
        self.annotations = data_list
        self.n_classes = len(self.annotations['target'].unique())
        self.n_domain = len(self.annotations['noise'].unique())
        self.transform = transform
        self.mode=mode
        to_tensor = []
        ## Resize the tensor here if needed
        to_tensor += [transforms.ToTensor()]
        # to_tensor += [transforms.Normalize(
        #     mean = (0.4914, 0.4822, 0.4465),
        #     std = (0.2470, 0.2435, 0.2616)) # (0.2470, 0.2435, 0.2616) (0.2023, 0.1994, 0.2010)
        #     ]
        to_tensor += [transforms.Normalize(
            mean = (0.485, 0.456, 0.406),
            std = (0.229, 0.224, 0.225)) # (0.2470, 0.2435, 0.2616) (0.2023, 0.1994, 0.2010)
            ]
       
        self.to_tensor = transforms.Compose(to_tensor)
        
        
    def __len__(self):
        return len(self.annotations)
        
    def __getitem__(self, index):
        data_path = self.annotations.iloc[index]['path'] 
        data = Image.open(data_path)
        #data = torch.load(data_path)
        if self.transform and self.mode=='train':
            data = self.transform(data)
        data = self.to_tensor(data)
        
        # data = torch.tensor(data) 
        # data = torch.permute(data,(2,0,1))
        label = torch.tensor(self.annotations.iloc[index]['target']) 
        domain = torch.tensor(self.annotations.iloc[index]['noise']) 
        sample = {'data': data, 'label': label, 'domain':domain}
        return sample
 

class DuplicateSampleTransform():
    
    def __init__(self,transform):
        self.transform = transform     
        
    def __call__(self,sample):
        x_i = self.transform(sample)
        x_j = self.transform(sample)
        return x_i, x_j
    
#sanity check for sampler:
# print(os.getcwd())
# #data_list = create_df_cifar10(filepath=args['data_dir'],noise=unseen_corruption,mode='test')
# test_loader = get_loader(args, mode='test',gamma=1, noise = ['normal'])
# data = next(iter(test_loader))
# label,domain = data['label'], data['domain']
# print(label.shape)
# for i in range(10):
#     #print(f'Label {i} => {(label==i).sum()}')
    
#     # print(f'Domain {domain}')
#     #print(f'Domain {i} => {(domain==i).sum()}')

def sanity_check():
    train_loader = get_loader(cfg, mode='test', corelated=False, noise = cfg.DATA.CORRUPTION_SET)
    data = next(iter(train_loader))
    data,label,domain = data['data'], data['label'], data['domain']
    print(label.shape)
    print(data.shape)
    print(domain)
    print(cfg.DATA.CORRUPTION_SET[domain[0].item()])

if __name__ == "__main__":
    sanity_check()