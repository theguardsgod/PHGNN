from torch import nn
from timm.models.layers import trunc_normal_
from typing import Sequence, Tuple, Union
from monai.networks.layers.utils import get_norm_layer
from monai.utils import optional_import
from .layers import LayerNorm
from .transformerblock import TransformerBlock, PromptTransformerBlock
from .dynunet_block import get_conv_layer, UnetResBlock
from .IntraAtt import TransformerEncoder
from .tab import Transformer, PromptTransformer, ConcatTransformer
import torch
import torch.nn.functional as F
import math
from torch.nn import Dropout
from functools import reduce
from operator import mul



einops, _ = optional_import("einops")

class UnetrPPEncoder(nn.Module):
    def __init__(self, input_size=[37 * 45 * 37, 18 * 22 * 18, 9 * 11 * 9, 4 * 5 * 4],dims=[32, 64, 128, 256],
                 proj_size =[64,64,64,32], depths=[3, 3, 3, 3],  num_heads=4, spatial_dims=3, in_channels=1,
                 dropout=0.2, transformer_dropout_rate=0.2 ,**kwargs):
        super().__init__()
        self.dims = dims
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        # stem_layer = nn.Sequential(
        #     get_conv_layer(spatial_dims, in_channels, dims[0], kernel_size=(3, 3, 3), stride=(3, 3, 3),
        #                    dropout=dropout, conv_only=True, ),
        #     get_norm_layer(name=("group", {"num_groups": in_channels}), channels=dims[0]),
        # )
        stem_layer = nn.Sequential(
            get_conv_layer(spatial_dims, in_channels, dims[0], kernel_size=(5, 5, 5), stride=(5, 5, 5),
                           dropout=dropout, conv_only=True, ),
            get_norm_layer(name=("group", {"num_groups": in_channels}), channels=dims[0]),
        )
        self.downsample_layers.append(stem_layer)
        for i in range(3):
            downsample_layer = nn.Sequential(
                get_conv_layer(spatial_dims, dims[i], dims[i + 1], kernel_size=(2, 2, 2), stride=(2, 2, 2),
                               dropout=dropout, conv_only=True, ),
                get_norm_layer(name=("group", {"num_groups": dims[i]}), channels=dims[i + 1]),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple Transformer blocks
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(TransformerBlock(input_size=input_size[i], hidden_size=dims[i],
                                                     proj_size=proj_size[i], num_heads=num_heads,
                                                     dropout_rate=transformer_dropout_rate, pos_embed=True))
            self.stages.append(nn.Sequential(*stage_blocks))
        self.hidden_states = []
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        # hidden_states = []
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        # hidden_states.append(x)

        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            # if i == 3:  # Reshape the output of the last stage
            #     x = einops.rearrange(x, "b c h w d -> b (h w d) c")
            # hidden_states.append(x)
        return x
        # return x, hidden_states

    def forward(self, x):
        img_features = self.forward_features(x)
        return img_features

class PromptUnetrPPEncoder(nn.Module):
    def __init__(self, input_size=[37 * 45 * 37, 18 * 22 * 18, 9 * 11 * 9, 4 * 5 * 4],dims=[32, 64, 128, 256],
                 proj_size =[64,64,64,32], depths=[3, 3, 3, 3],  num_heads=4, spatial_dims=3, in_channels=1,
                 dropout=0.2, transformer_dropout_rate=0.2 ,**kwargs):
        super().__init__()
        self.dims = dims
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem_layer = nn.Sequential(
            get_conv_layer(spatial_dims, in_channels, dims[0], kernel_size=(3, 3, 3), stride=(3, 3, 3),
                           dropout=dropout, conv_only=True, ),
            get_norm_layer(name=("group", {"num_groups": in_channels}), channels=dims[0]),
        )
        self.downsample_layers.append(stem_layer)
        for i in range(3):
            downsample_layer = nn.Sequential(
                get_conv_layer(spatial_dims, dims[i], dims[i + 1], kernel_size=(2, 2, 2), stride=(2, 2, 2),
                               dropout=dropout, conv_only=True, ),
                get_norm_layer(name=("group", {"num_groups": dims[i]}), channels=dims[i + 1]),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple Transformer blocks
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                if i < 3:
                    stage_blocks.append(TransformerBlock(input_size=input_size[i], hidden_size=dims[i],
                                                     proj_size=proj_size[i], num_heads=num_heads,
                                                     dropout_rate=transformer_dropout_rate, pos_embed=True))
                else:
                    stage_blocks.append(PromptTransformerBlock(input_size=input_size[i], hidden_size=dims[i],
                                                     proj_size=proj_size[i], num_heads=num_heads,
                                                     dropout_rate=transformer_dropout_rate, pos_embed=True))
            self.stages.append(nn.Sequential(*stage_blocks))
        self.hidden_states = []
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def train(self, mode=True):
        if mode:
            self.downsample_layers.eval()
            self.stages[0].eval()
            self.stages[1].eval()
            self.stages[2].eval()
            self.stages[3].train()
            print("PromptUnetrPPEncoder train mode")
        else:
            # eval:
            for module in self.children():
                module.train(mode)



    def forward_features(self, x):
        # hidden_states = []
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        # hidden_states.append(x)

        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            # if i == 3:  # Reshape the output of the last stage
            #     x = einops.rearrange(x, "b c h w d -> b (h w d) c")
            # hidden_states.append(x)
        return x
        # return x, hidden_states

    def forward(self, x):
        img_features = self.forward_features(x)
        return img_features



class UnetrPP(nn.Module):
    def __init__(self, input_size=[37 * 45 * 37, 18 * 22 * 18, 9 * 11 * 9, 4 * 5 * 4],dims=[32, 64, 128, 256],
                 proj_size =[64,64,64,32], depths=[3, 3, 3, 3],  num_heads=8, spatial_dims=3, in_channels=1,
                 dropout=0.2, transformer_dropout_rate=0.2 ,**kwargs):
        super().__init__()
        self.dims = dims

        self.img_encoder = UnetrPPEncoder(
                        input_size=input_size,
                        dims=dims, 
                        depths=depths, 
                        num_heads=num_heads,
                        in_channels=in_channels,
                        proj_size = proj_size,
                        spatial_dims=spatial_dims,
                        dropout=dropout,
                        transformer_dropout_rate=transformer_dropout_rate
                        )

        self.sig  = nn.Sigmoid()

        # self.mlp_head = nn.Sequential(
        #     nn.Linear(dims[3], 512),
        #     nn.LayerNorm(512),
        #     nn.Linear(512, 1)
        # )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dims[3]),
            nn.Linear(dims[3], 1)
        )
        self.sig = nn.Sigmoid()

        #self.avgpool = nn.AdaptiveAvgPool3d(1)

    def train(self, mode=True):
        if mode:
            # training:

            self.img_encoder.train()

        else:
            # eval:
            for module in self.children():
                module.train(mode) 
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x, clinical):
        x = self.img_encoder(x)
        # avg pooling
        x = x.mean(dim = [2,3,4])
        x = self.mlp_head(x)
        x = self.sig(x)
        
        return x



class thenet(nn.Module):
    def __init__(self, input_size=[37 * 45 * 37, 18 * 22 * 18, 9 * 11 * 9, 4 * 5 * 4],dims=[32, 64, 128, 256],
                 proj_size =[64,64,64,32], depths=[3, 3, 3, 3],  num_heads=8, spatial_dims=3, in_channels=1,
                 dropout=0.2, transformer_dropout_rate=0.2 ,**kwargs):
        super().__init__()
        self.dims = dims

        self.img_encoder = PromptUnetrPPEncoder(
                        input_size=input_size,
                        dims=dims, 
                        depths=depths, 
                        num_heads=num_heads,
                        in_channels=in_channels,
                        proj_size = proj_size,
                        spatial_dims=spatial_dims,
                        dropout=dropout,
                        transformer_dropout_rate=transformer_dropout_rate
                        )

        self.sig  = nn.Sigmoid()
        self.classifier1 = nn.Linear(dims[3], 1)

        self.tabformer = PromptTransformer(
            num_tokens = 9,                  # number of features, paper set at 512   调整特征数要调整这里
            dim = 256,                           # dimension, paper set at 512
            dim_head= 16,
            depth = 3,                          # depth, paper recommended 3
            heads = 8,                          # heads, paper recommends 8
            attn_dropout = 0.1,                 # post-attention dropout
            ff_dropout = 0.1,                   # feed forward dropout
            prompt_dropout = 0.1,
            prompt_num_tokens = 100,
            prompt_dim = 512, 
            vis = False
        )

        self.ConcatTransformer = ConcatTransformer(
            dim = 256, 
            depth = 3, 
            heads = 8, 
            dim_head = 16, 
            attn_dropout = 0.1, 
            ff_dropout = 0.1, 
            vis = False, 
            vis_patch_num = 80, 
            tab_patch_num = 9, 
            emb_dropout = 0.1, 
            num_classes = 1
        )


        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def train(self, mode=True):
        if mode:
            # training:
            self.tabformer.train()
            self.img_encoder.train()
            self.ConcatTransformer.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode) 
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x, clinical):
        img_features = self.img_encoder(x)
        B,C,H,W,D = img_features.shape 
        img_features = img_features.view(B,C,-1).permute(0,2,1)
        clinical = self.tabformer(clinical.unsqueeze(2))
        clinical = clinical[:, -9:, :]
        fuse_features = torch.cat((img_features, clinical), dim=1)
        out = self.ConcatTransformer(fuse_features)
        
        return out

