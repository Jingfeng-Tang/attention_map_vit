import torch
import torch.nn as nn
from functools import partial
# from model.vit import VisionTransformer, _cfg
from model.vision_transformer import VisionTransformer, _create_vision_transformer
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import torch.nn.functional as F

import math


__all__ = [
    'vit_small_patch16_224_attmap',
    'vit_tiny_patch16_224_attmap',
    'vit_base_patch16_224_attmap'
]

class Attmap(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.head = nn.Linear(self.embed_dim, 200)
        # self.head.apply(self._init_weights)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        # print(self.training)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - self.num_classes
        N = self.pos_embed.shape[1] - self.num_classes
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0:self.num_classes]
        patch_pos_embed = self.pos_embed[:, self.num_classes:]
        dim = x.shape[-1]

        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward_features(self, x, n=12):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        # a = []
        # b = a[1]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.pos_drop(x)
        attn_weights = []

        for i, blk in enumerate(self.blocks):
            # print('ok---------')
            x, weights_i = blk(x)
            # print(f'x: {x.shape}')  # [128, 197, 384]
            # print(f'weights_i: {weights_i.shape}')  # [128, 6, 197, 197]
            attn_weights.append(weights_i)

        return x[:, 0:1], x[:, 1:], attn_weights

    def forward(self, x, return_att=False, n_layers=12, attention_type='a'):
        w, h = x.shape[2:]
        x_cls, x_patch, attn_weights = self.forward_features(x)
        # print(f'x_cls: {x_cls.shape}')  # [128, 1, 384]
        # print(f'x_patch: {x_patch.shape}')  # [128, 196, 384]
        # print(f'attn_weights: {len(attn_weights)}')  # 12
        # a = []
        # b = a[1]
        n, p, c = x_patch.shape
        if w != h:
            w0 = w // self.patch_embed.patch_size[0]
            h0 = h // self.patch_embed.patch_size[0]
            x_patch = torch.reshape(x_patch, [n, w0, h0, c])
        else:
            x_patch = torch.reshape(x_patch, [n, int(p ** 0.5), int(p ** 0.5), c])
        x_patch = x_patch.permute([0, 3, 1, 2])
        x_patch = x_patch.contiguous()

        attn_weights = torch.stack(attn_weights)  # 12 * B * H * N * N

        if return_att:
            if attention_type == 'a':
                attn_weights = torch.mean(attn_weights, dim=2)  # 12 * B * N * N
                n, c, h, w = x_patch.shape
                att_w = attn_weights[-n_layers:].sum(0)[:, 0:1, 1:].reshape([n, h, w])
                # print(f'att_w: {att_w.shape}')  # [128, 14, 14]
                att_map = att_w  # B * C * 14 * 14
            else:
                patch_attn = attn_weights[:, :, 1:, 1:]

        x_cls = x_cls.squeeze(1)
        x_cls_logits = self.head(x_cls)

        if return_att:
            return x_cls_logits, att_map
        else:
            return x_cls_logits


@register_model
def vit_small_patch16_224_attmap(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    """ ViT-Small (ViT-S/16)
    """
    model = Attmap(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@register_model
def vit_tiny_patch16_224_attmap(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs) -> VisionTransformer:
    """ ViT-Tiny (Vit-Ti/16)
    """
    model = Attmap(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@register_model
def vit_base_patch16_224_attmap(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs) -> VisionTransformer:
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model = Attmap(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

