import torch
from torch import nn
from transformers import LlamaForCausalLM, Cache, DynamicCache
from transformers.activations import GELUActivation
from transformers.utils import logging

from .configuration_live_llama import LiveLlamaConfig
from ..modeling_live import build_live, LiveMixin
import random
logger = logging.get_logger(__name__)

class LiveLlamaForCausalLM(LlamaForCausalLM, LiveMixin):
    config_class = LiveLlamaConfig
    _keys_to_ignore_on_load_missing = ['vision_encoder', 'connector']

    def __init__(self, config: LiveLlamaConfig):
        super().__init__(config)

    def preprocessInputForMEM(self, tensors):
        bs = 1
        sample_n = min(len(tensors[0]), 10240 // len(tensors[0][0]))
        indices = torch.randperm(tensors[0].size(0))[:sample_n]
        
        return [i[indices] for i in tensors]

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        frames: torch.FloatTensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_values: list[torch.FloatTensor] = None,
        inputs_embeds: torch.FloatTensor = None,
        labels: torch.LongTensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        cache_position: torch.LongTensor = None,
        frames_lens = None,
        **kwargs,
    ):
        if isinstance(past_key_values, (tuple, list)):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        if inputs_embeds is None:
            _, inputs_embeds, labels, attention_mask, position_ids = self.joint_embed(frames, frames_lens, **kwargs)
        if attention_mask is not None and attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.to(dtype=inputs_embeds.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(inputs_embeds.dtype).min
        
        outputs = super().forward(
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            # labels
            use_cache = use_cache,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
            cache_position=cache_position,
        )
        loss = None
        if labels is not None:
            logits = outputs[0]
            loss = nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction='none')
            loss = loss.sum() / (labels >= 0).sum()

        # if torch.isnan(loss).any():
        #     print("loss is nan!!!!!")
        # if loss is None:
        #     print("loss is None!!!!!")
        if not return_dict:
            return (loss,) + outputs[1:] if loss is not None else outputs
        outputs.loss = loss
        return outputs

def build_live_llama(**kwargs):
    return build_live(config_class=LiveLlamaConfig, model_class=LiveLlamaForCausalLM, **kwargs)

if __name__ == '__main__':
    from ..arguments_live import LiveMemoryTrainingArguments
    print(LiveMemoryTrainingArguments().to_dict())
    model, tokenizer = build_live_llama(is_training=True, **LiveMemoryTrainingArguments().to_dict())
    print(model.config, tokenizer)
