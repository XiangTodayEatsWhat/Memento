import math, torch
from functools import partial
from torch import nn, Tensor
from torchvision.transforms.functional import normalize
from transformers import AutoModel
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

from .configuration_live import LiveConfigMixin

import torch.nn.functional as F

class NeuralTuringMachine(nn.Module):
    def __init__(self, input_dim=1024, output_dim=1024, attention_dropout=0.1):
        super(NeuralTuringMachine, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.q_proj = nn.Linear(input_dim, output_dim, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(input_dim, output_dim, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(input_dim, output_dim, dtype=torch.bfloat16)
        self.dropout = nn.Dropout(attention_dropout)
        self.out_proj = nn.Linear(output_dim, 1, dtype=torch.bfloat16)
        self.out_dropout = nn.Dropout(attention_dropout)
        self.out_act = nn.Sigmoid()
        # self.out_ln = nn.LayerNorm(input_dim, eps=1e-12, dtype=torch.bfloat16)

    # def get_weight(self, x, y):
    #     query = self.q_proj(x)
    #     key = self.k_proj(y)
    #     scores = torch.matmul(query, key.transpose(0, 1)) / math.sqrt(self.output_dim)
    #     weight = F.softmax(scores, dim=-1)
    #     return weight
    
    # def get_output(self, x, y):
    #     query = self.q_proj(x)
    #     key = self.k_proj(y)
    #     scores = torch.matmul(query, key.transpose(0, 1)) / math.sqrt(self.output_dim)
    #     weight = F.softmax(scores, dim=-1)
    #     attn = self.dropout(weight)
    #     value = self.v_proj(y)
    #     output = torch.matmul(attn, value)
    #     output = self.out_proj(output)
    #     output = self.out_dropout(output)
    #     output = self.out_ln(output.unsqueeze(0)).squeeze(0)
    #     return output
    

    def forward(self, x, y, type="scores"):
        query = self.q_proj(x)
        key = self.k_proj(y)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.output_dim)
        if type == "scores":
            return scores
        elif type == "attn":
            weight = F.softmax(scores, dim=-1)
            attn = self.dropout(weight)
            value = self.v_proj(y)
            output = torch.matmul(attn, value)
            output = self.out_proj(output)
            output = self.out_dropout(output)
            output = self.out_act(output.sum(-1).sum(-1))
            # output = self.out_ln(output.unsqueeze(0)).squeeze(0)
            return output
            

       
class MultiNeuralTuringMachine(nn.Module):
    def __init__(self, text_input_dim=1024, visual_input_dim=1024, output_dim=1024, attention_dropout=0.1):
        super(MultiNeuralTuringMachine, self).__init__()
        self.output_dim = output_dim
        self.q_proj = nn.Linear(visual_input_dim, output_dim, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(text_input_dim, output_dim, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(text_input_dim, output_dim, dtype=torch.bfloat16)
        self.dropout = nn.Dropout(attention_dropout)
        self.out_proj = nn.Linear(output_dim, 1, dtype=torch.bfloat16)
        self.out_dropout = nn.Dropout(attention_dropout)
        self.out_act = nn.Sigmoid()
    
    def forward(self, x, y):
        query = self.q_proj(x)
        key = self.k_proj(y)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.output_dim)
        weight = F.softmax(scores, dim=-1)
        attn = self.dropout(weight)
        value = self.v_proj(y)
        output = torch.matmul(attn, value)
        output = self.out_proj(output)
        output = self.out_dropout(output)
        output = self.out_act(output.sum(-1).sum(-1))
        return output
 

def _siglip_vision_encode(vision_model: nn.Module, frames: Tensor, frame_token_cls: bool, frame_token_pooled: tuple,
    mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], rescale_factor=0.00392156862745098, **kwargs):
    frames = normalize(frames * rescale_factor, mean=mean, std=std)
    with torch.cuda.amp.autocast():
        vision_outputs = vision_model(frames)
        last_hidden_state = vision_outputs.last_hidden_state
        if frame_token_pooled:
            s = int(math.sqrt(last_hidden_state.shape[1]))
            # print(s,frame_token_pooled, last_hidden_state.shape)
            spatial_tokens = torch.nn.functional.adaptive_avg_pool2d(
                last_hidden_state.reshape(
                    last_hidden_state.shape[0], s, s, last_hidden_state.shape[-1]
                ).permute(0, 3, 1, 2),
                frame_token_pooled
            ).flatten(2, 3).permute(0, 2, 1)
            if not frame_token_cls:
                return spatial_tokens
        if frame_token_cls:
            cls_token = vision_outputs.pooler_output[:, None]
            if not frame_token_pooled:
                return cls_token
    return torch.cat([cls_token, spatial_tokens], dim=1)

def _clip_vision_encode(vision_model: nn.Module, frames: Tensor, frame_token_cls: bool, frame_token_pooled: tuple,
    mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, rescale_factor=0.00392156862745098, **kwargs):
    frames = normalize(frames * rescale_factor, mean=mean, std=std)
    with torch.cuda.amp.autocast():
        vision_outputs = vision_model(frames)
        last_hidden_state = vision_outputs.last_hidden_state
        if frame_token_pooled:
            s = int(math.sqrt(last_hidden_state.shape[1]))
            spatial_tokens = torch.nn.functional.adaptive_avg_pool2d(
                last_hidden_state[:,1:].reshape(
                    last_hidden_state.shape[0], s, s, last_hidden_state.shape[-1]
                ).permute(0, 3, 1, 2),
                frame_token_pooled
            ).flatten(2, 3).permute(0, 2, 1)
            if not frame_token_cls:
                return spatial_tokens
        if frame_token_cls:
            cls_token = last_hidden_state[:,0]
            if not frame_token_pooled:
                return cls_token
    return torch.cat([cls_token, spatial_tokens], dim=1)

def build_live_vision(config: LiveConfigMixin):
    model = AutoModel.from_pretrained(config.vision_pretrained).vision_model
    ######
    if 'siglip-large-patch16-384' in config.vision_pretrained:
    # if 'google/siglip-large-patch16-384' == config.vision_pretrained:
    ######
        return model, partial(_siglip_vision_encode, frame_token_cls=config.frame_token_cls, frame_token_pooled=config.frame_token_pooled)
    elif 'laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90k' == config.vision_pretrained or 'openai/clip-vit-large-patch14-336' == config.vision_pretrained:
        return model, partial(_clip_vision_encode, config)
    else:
        raise ValueError(f'Unverified vision_pretrained: {config.vision_pretrained}')