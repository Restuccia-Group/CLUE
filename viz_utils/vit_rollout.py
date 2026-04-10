from timm.models.layers import PatchEmbed
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

import torch
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np

# def rollout(attentions, discard_ratio, head_fusion):
#     result = torch.eye(attentions[0].size(-1))
#     with torch.no_grad():
#         for attention in attentions:
#             if head_fusion == "mean":
#                 attention_heads_fused = attention.mean(axis=1)
#             elif head_fusion == "max":
#                 attention_heads_fused = attention.max(axis=1)[0]
#             elif head_fusion == "min":
#                 attention_heads_fused = attention.min(axis=1)[0]
#             else:
#                 raise "Attention head fusion type Not supported"

#             # Drop the lowest attentions, but
#             # don't drop the class token
#             # print(attention_heads_fused.size())
#             attention_heads_fused = attention_heads_fused.squeeze()
#             flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
#             _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
#             indices = indices[indices != 0]
#             flat[0, indices] = 0

#             I = torch.eye(attention_heads_fused.size(-1))
#             print(I.size(), attention_heads_fused.size())
#             a = (attention_heads_fused + 1.0*I)/2
#             a = a / a.sum(dim=-1)

#             result = torch.matmul(a, result)
    
#     # Look at the total attention between the class token,
#     # and the image patches
#     print(result.size())
#     mask = result[0, 0 , 1 :]
#     # In case of 224x224 image, this brings us from 196 to 14
#     width = int(mask.size(-1)**0.5)
#     mask = mask.reshape(width, width).numpy()
#     mask = mask / np.max(mask)
#     return mask    

# # def get_attention(attentions):
# #         def _hook(module, input, output):
# #             print("Got here")
# #             attentions.append(output.detach().clone().cpu())

# #         return _hook
#         # self.attentions.append(output.detach().clone().cpu())

# class VITAttentionRollout:
#     def __init__(self, model, attention_layer_name='attn_proj', head_fusion="mean",
#         discard_ratio=0.9):
#         self.model = model
#         self.head_fusion = head_fusion
#         self.discard_ratio = discard_ratio
#         self.handles = []
#         self.attn_layer = attention_layer_name
#         self.attentions = []
#         # for name, module in self.model.named_modules():
#         #     if attention_layer_name in name:
#         #         print(name)
#         #         self.handles.append(module.register_forward_hook(self.get_attention))

#         # self.handles.append(self.model.blocks[11].attn.attn_drop.register_forward_hook(self.get_attention))

        

#     # def get_attention(self, module, input, output):
#     #     # def _hook(module, input, output):
#     #     #     print("Got here")
#     #     #     module.attention = output.detach().clone().cpu()

#     #     # return _hook
#     #     self.attentions.append(output.detach().clone().cpu())

#     def get_attention(self, module, input, output):
#         # def _hook(module, input, output):
#         print("Got here")
#         self.attentions.append(output.detach().clone().cpu())

#         # return _hook

#     def __call__(self, input_tensor):
#         self.attentions = []
#         # self.model.train()

#         # This is the "one line of code" that does what you want
#         feature_extractor = create_feature_extractor(
#                     self.model, return_nodes=[f'blocks.{N}.attn.scaled_dot_product_attention' for N in range(12)],
#                     tracer_kwargs={'leaf_modules': [PatchEmbed]})
#         # print(self.model.training)
#         with torch.no_grad():
#             # output = self.model(input_tensor)
#             out = feature_extractor(input_tensor)


#         print(out['blocks.11.attn.scaled_dot_product_attention'].shape)
#         print(out['blocks.0.attn.scaled_dot_product_attention'].shape)
#         for N in range(12):
#             self.attentions.append(out[f'blocks.{N}.attn.scaled_dot_product_attention'])
#         print('Length : ', len(self.attentions), len(self.handles))
#         # raise Exception

#         # self.model.eval()
#         return rollout(self.attentions, self.discard_ratio, self.head_fusion)


def rollout(attentions, discard_ratio, head_fusion):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)
    
    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0 , 1 :]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask    

class VITAttentionRollout:
    def __init__(self, model, attention_layer_name='attn_drop', head_fusion="mean",
        discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)

        self.attentions = []

        for block in self.model.blocks:
            block.attn.fused_attn = False

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor)

        return rollout(self.attentions, self.discard_ratio, self.head_fusion)