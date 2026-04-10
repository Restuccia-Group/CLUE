import argparse
import sys
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

from viz_utils.vit_rollout import VITAttentionRollout
from viz_utils.vit_grad_rollout import VITAttentionGradRollout

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image_path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--head_fusion', type=str, default='max',
                        help='How to fuse the attention heads for attention rollout. \
                        Can be mean/max/min')
    parser.add_argument('--discard_ratio', type=float, default=0.9,
                        help='How many of the lowest 14x14 attention paths should we discard')
    parser.add_argument('--category_index', type=int, default=None,
                        help='The category index for gradient rollout')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU")
    else:
        print("Using CPU")

    return args

def show_mask_on_image(img, mask, alpha=0.3):
    img = np.float32(img) #/ 255
    print("Shape of image : ", img.shape)
    colormap = plt.get_cmap('jet')
    # Apply the colormap, which outputs an RGBA image in 0-1 range
    # print(np.max(mask), np.min(mask))
    heatmap = colormap(mask)
    print("Shape of heatmap : ", heatmap.shape)
    # heatmap = np.transpose(heatmap, (2, 0, 1))
    # print(type(heatmap), heatmap.shape)
    # heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) #/ 255
    # print(np.max(heatmap), np.min(heatmap))
    # cam = heatmap[:,:,:3] + np.float32(img)
    cam = (1 - alpha) * np.float32(img) + alpha * heatmap[:,:,:3]  # Blend with alpha
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def attention_rollout(model, loader, args=None, num_classes=10, num_image_per_class=5, category_index=None):
    device = next(model.parameters())[0].device
    # class_img_pair = {class_id:[] for class_id in range(num_classes)}
    # for img, label in loader:
    #     flag = True
    #     for l in label:
    #         if len(class_img_pair[l.item()])<num_image_per_class:
    #             class_img_pair[l.item()].append(img[l.item()].to(device))
    #             flag = False

    #     if flag:
    #         break

    # Initialize a dictionary to hold images for each class
    class_img_pair = {class_id:[] for class_id in range(num_classes)}

    denormalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
        )
    # Iterate through the DataLoader
    for images, labels in loader:
        flag = True
        for img, label in zip(images, labels):

            # Convert to numpy and add to the class_images dictionary
            if label is None or torch.isnan(label):
                label = model(img.unsqueeze(0).to(device))
                label = label.argmax(1)
            if len(class_img_pair[label.item()])<num_image_per_class: 
                class_img_pair[label.item()].append(img.to(device))

            # Stop collecting images if we have enough for each class
            if sum(len(class_img_pair[class_label]) == num_image_per_class for class_label in class_img_pair)>=10:
                flag = False
                break

        if not flag:
            break
    

    for key in class_img_pair:
        img_arr = class_img_pair[key]
        if len(img_arr)==0:
            continue
        for ind, img in enumerate(img_arr):
            if category_index is None:
                print("Doing Attention Rollout")
                attention_rollout = VITAttentionRollout(model, head_fusion='mean', 
                    discard_ratio=0.6)
                mask = attention_rollout(img.unsqueeze(0))
                name = "attention_rollout_{0.9}_{max}.png"
            else:
                print("Doing Gradient Attention Rollout")
                grad_rollout = VITAttentionGradRollout(model, discard_ratio=0.8)
                mask = grad_rollout(img.unsqueeze(0), int(key))
                name = "grad_rollout_{}_{:.3f}_{}.png".format(int(key),
                    0.9, 'max')
                
            img = denormalize(img)
            img = torch.clip(img, min=0.0, max=1.0)
            np_img = np.array(img.detach().clone().cpu().numpy())  #We do not need to change RGB to BGR for plt[:, :, ::-1]
            np_img = np.transpose(np_img, (1, 2, 0))

            pil_image = Image.fromarray(mask)
            mask = pil_image.resize((np_img.shape[1], np_img.shape[0]), Image.BILINEAR)
            # mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
            mask = np.array(mask)
            # print("Mask Shape : ", mask.shape)
            # print("Image Shape : ", np_img.shape)
            # print(np.max(np_img), np.min(np_img))
            # raise Exception
            masked_img = show_mask_on_image(np_img, mask)
            # pil_image = Image.fromarray(masked_img)
            # masked_img = pil_image.resize((256, 256), Image.BILINEAR)
            # masked_img = np.array(masked_img)

            # pil_image = Image.fromarray(np_img)
            # np_img = pil_image.resize((128, 128), Image.BILINEAR)

            #Showing the image
            # np_img = np_img[:,:,::-1]
            # print(np_img.shape)
            # plt.imshow("Input Image", np_img)
            # plt.imshow(name, mask)
            
            plt.imsave(f"./imagenet_viz_assets/class_id_{int(key)}_{ind}_input.png", np_img)
            plt.imsave(f"./imagenet_viz_assets/class_id_{int(key)}_{ind}_attn_rollout.png", masked_img)

        

# if __name__ == '__main__':
#     args = get_args()
#     #Will need to change the model here
#     model = torch.hub.load('facebookresearch/deit:main', 
#         'deit_tiny_patch16_224', pretrained=True)
#     model.eval()

#     if args.use_cuda:
#         model = model.cuda()

#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#     ])
#     img = Image.open(args.image_path)
#     img = img.resize((224, 224))
#     input_tensor = transform(img).unsqueeze(0)
#     if args.use_cuda:
#         input_tensor = input_tensor.cuda()

#     if args.category_index is None:
#         print("Doing Attention Rollout")
#         attention_rollout = VITAttentionRollout(model, head_fusion=args.head_fusion, 
#             discard_ratio=args.discard_ratio)
#         mask = attention_rollout(input_tensor)
#         name = "attention_rollout_{:.3f}_{}.png".format(args.discard_ratio, args.head_fusion)
#     else:
#         print("Doing Gradient Attention Rollout")
#         grad_rollout = VITAttentionGradRollout(model, discard_ratio=args.discard_ratio)
#         mask = grad_rollout(input_tensor, args.category_index)
#         name = "grad_rollout_{}_{:.3f}_{}.png".format(args.category_index,
#             args.discard_ratio, args.head_fusion)


#     np_img = np.array(img)  #We do not need to change RGB to BGR for plt[:, :, ::-1]
#     pil_image = Image.fromarray(mask)
#     mask = mask.resize((np_img.shape[1], np_img.shape[0]), Image.BILINEAR)
#     # mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
#     mask = show_mask_on_image(np_img, mask)

#     #Showing the image
#     plt.imshow("Input Image", np_img)
#     plt.imshow(name, mask)
#     plt.imsave("./input.png", np_img)
#     plt.imsave(name, mask)
