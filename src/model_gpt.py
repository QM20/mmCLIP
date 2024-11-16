import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoProcessor
import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from functools import partial
from PIL import Image
import sys
sys.path.append('.')
from lib.VisionTransformer import VisionTransformer, ViT_wo_patch_embed, MB_ViT_v3, MB_ViT_v3_shareweight
from timm.models.vision_transformer import VisionTransformer as timm_vit
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()


class mmCLIP_gpt_multi_brach_property_v3(nn.Module):##implemented as paper
    def __init__(self, proj_head_dim=64, if_use_hm_proj=False, if_use_text_proj=False, if_use_hm_att=True,
                 if_use_text_att=True, if_use_hm=True, device=None, in_channels=3):
        super().__init__()
        if if_use_hm:
            self.heatmap_encoder=MB_ViT_v3()
            # self.heatmap_encoder=MB_ViT_v3_shareweight()
        else:
            assert NotImplementedError
        self.if_use_hm_attn = if_use_hm_att
        self.if_use_text_attn = if_use_text_att
        if self.if_use_hm_attn:
            self.hm_self_attention = ViT_wo_patch_embed(global_pool=False, embed_dim=128*3, depth=1,
                                                  num_heads=4, mlp_ratio=4, qkv_bias=True,
                                                  norm_layer=partial(nn.LayerNorm, eps=1e-6))  # in: B*L*C
            self.hm_attn_proj = nn.Sequential(nn.Linear(128*3, 512))
        if self.if_use_text_attn:
            self.text_self_attention = ViT_wo_patch_embed(global_pool=False, embed_dim=512, depth=1,
                                                  num_heads=4, mlp_ratio=4, qkv_bias=True,
                                                  norm_layer=partial(nn.LayerNorm, eps=1e-6))  # in: B*L*C
            # self.text_attn_proj = nn.Sequential(nn.Linear(512, 512))
        self.if_use_hm_proj = if_use_hm_proj
        self.if_use_text_proj = if_use_text_proj
        if if_use_hm_proj:
            self.hm_proj = nn.Sequential(nn.Linear(512, proj_head_dim))
        if if_use_text_proj:
            self.text_proj = nn.Sequential(nn.Linear(512,proj_head_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.clip_encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").requires_grad_(False)
        for param in self.clip_encoder.parameters():
            param.requires_grad = False
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32")  # openai/clip-vit-large-patch14
        self.device = device

    def cal_text_features_2d(self, text_list_2d):  # B*k
        length = len(text_list_2d)
        text_branches = len(text_list_2d[0])

        text_list = list(np.array(text_list_2d).reshape(-1))
        text_input = self.clip_processor(text=text_list, return_tensors="pt", padding=True).to(self.device)
        text_embeds = self.clip_encoder.get_text_features(**text_input)
        text_embeds = (text_embeds.reshape(length, text_branches, -1))
        if self.if_use_text_attn:
            text_embeds_att, _ = self.text_self_attention(text_embeds)
        else:
            text_embeds_att = text_embeds.mean(dim=1)
        text_embeds = torch.cat([text_embeds, text_embeds_att.unsqueeze(1)], dim=1)
        # text_embeds=self.text_attn_proj(text_embeds)
        if self.if_use_text_proj:
            text_embeds = self.text_proj(text_embeds)
        return text_embeds

    def cal_hm_features(self, hm_input):
        hm_embeds =self.heatmap_encoder(hm_input)
        out, _=self.hm_self_attention(hm_embeds)
        result=torch.cat([hm_embeds, out.unsqueeze(1)], dim=1)
        hm_embeds=self.hm_attn_proj(result)
        if self.if_use_hm_proj:
            hm_embeds = self.hm_proj(hm_embeds)
        return hm_embeds, None





