from .ResNet import *
from .ResNets import *
from .VGG import *
from .VGG_LTH import *
from .Swin import *
from .transformers import *

model_dict = {
    "resnet18": resnet18,   #Imagenet arch
    "resnet50": resnet50,   #Imagenet arch
    "resnet20s": resnet20s, #CIFAR-arch
    "resnet44s": resnet44s, #CIFAR-arch
    "resnet56s": resnet56s, #CIFAR-arch
    "vgg16_bn": vgg16_bn,   #Common-arch
    "vgg16_bn_lth": vgg16_bn_lth,   #Common-arch
    "swin_t":swin_t,        #CIFAR-arch
    "swin_s":swin_s,        #CIFAR-arch
    "swin_b":swin_b,        #CIFAR-arch
    "swin_l":swin_l,        #CIFAR-arch
    "vit_t":vit_t,          #CIFAR-arch
    "vit_b":vit_b,          #CIFAR-arch
    "vit_b_16":vit_b_16,    #Imagenet arch
    "vit_b_32":vit_b_32,    #Imagenet arch
    "vit_l_32":vit_l_32,    #Imagenet arch
    "vit_l_16":vit_l_16,    #Imagenet arch
    "vit_h_14":vit_h_14,    #Imagenet arch
    "swin_v2_t":swin_v2_t,  #Imagenet arch
    "swin_v2_s":swin_v2_s,  #Imagenet arch
    "swin_v2_b":swin_v2_b,  #Imagenet arch

}
