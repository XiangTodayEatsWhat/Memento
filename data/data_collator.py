import torch
from functools import partial
from transformers import PreTrainedTokenizer


def data_collator(batch: list[list], *, tokenizer: PreTrainedTokenizer, **kwargs):
    batch = list(zip(*batch))
    batch_plan, batch_frames, batch_frame_lens, batch_user_frames, batch_user_qs, batch_sample_idx, batch_evaluation_kwargs = batch
    batch = dict()
    batch['plans'] = batch_plan
    batch['frames'] = torch.stack(batch_frames)
    batch['frames_lens'] = batch_frame_lens
    batch["user_frames"] = batch_user_frames
    batch["user_qs"] = batch_user_qs
    batch['sample_idxs'] = torch.tensor(batch_sample_idx)
    batch['tokenizer'] = tokenizer
    batch['system_prompt'] = kwargs.get('system_prompt_mem', kwargs.get('system_prompt', ''))
    if batch_evaluation_kwargs and batch_evaluation_kwargs[0]:
        batch['evaluation_kwargs'] = batch_evaluation_kwargs[0]
    batch["qms_ratio"] = kwargs.get('qms_ratio', 0.5)
    return batch


def get_data_collator(**kwargs):
    return partial(data_collator, **kwargs)
