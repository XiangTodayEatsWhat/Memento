import torch, os
from peft import LoraConfig, get_peft_model, PeftModel
from transformers.activations import GELUActivation
from transformers import AutoModelForCausalLM, Cache
from transformers.utils import logging

from .tokenization_live import build_live_tokenizer_and_update_config, build_live_tokenizer_and_update_config_src
from .vision_live import build_live_vision, NeuralTuringMachine, MultiNeuralTuringMachine

import torch.nn.functional as F
from transformers.trainer_pt_utils import LabelSmoother


logger = logging.get_logger(__name__)

class LiveMixin(AutoModelForCausalLM):
    def set_vision_inside(self):
        logger.warning_once("!!! Set vision encoder in the model, only recommended for on in-the-wild inference. "
            "Please dont call this for efficient training & evaluation. Instead, do visual feature pre-extraction.")
        self.vision_encoder, self.vision_encode = build_live_vision(self.config)
    
    def set_connector_inside(self):
        self.connector = torch.nn.Sequential(
            torch.nn.Linear(self.config.vision_hidden_size, self.config.hidden_size, bias=True, dtype=torch.bfloat16),
            GELUActivation(self.config.hidden_size),
            torch.nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=True, dtype=torch.bfloat16),
        )

    def set_attention_inside(self):
        self.visual_attention_model = NeuralTuringMachine(self.config.vision_hidden_size, 
                                                          self.config.vision_hidden_size)
        self.multi_attention_model = MultiNeuralTuringMachine(self.config.hidden_size, 
                                                              self.config.vision_hidden_size, 
                                                              self.config.vision_hidden_size)


    def unset_vision_inside(self):
        del self.vision_encoder
        del self.vision_encode
    
    def attention_merge(self, turing_memory, new_feature, scores, update_ratio=None):
        if update_ratio is None:
            update_ratio = getattr(self, "attention_update_ratio", 0.2)
        weight = scores.softmax(-1)
        weight = weight * update_ratio
        decay = weight.sum(dim=1, keepdim=True)
        turing_memory = turing_memory * (1 - decay) + torch.mm(weight, new_feature)
        return turing_memory

    def mem_select(self, RFMemory, frame_i, qms_ratio=1.0, tokenizer=None):
        while len(self.user_frames) > 0 and frame_i + 1 >= self.user_frames[0]:
            _, user_q = self.user_frames[0], self.user_qs[0] + "\n"
            self.user_frames = self.user_frames[1:]
            self.user_qs = self.user_qs[1:]
            user_ids = tokenizer(user_q, return_tensors="pt", add_special_tokens=False).input_ids.to(self.model.device)
            user_embeds = self.get_input_embeddings()(user_ids.clamp(max=self.vocab_size-1))
            self.history_q = user_embeds if self.history_q is None else torch.cat([self.history_q, user_embeds], dim=1)
        if self.history_q is not None and len(RFMemory) >= 2:
            Memory = torch.stack(RFMemory[:-1])
            corr = self.multi_attention_model(Memory, self.history_q)
            topk_indices = torch.topk(corr, k=max(int(qms_ratio * len(corr)), 1)).indices
            Memory = Memory[topk_indices] * corr[topk_indices].view(-1, 1, 1)
            return torch.cat([Memory, RFMemory[-1].unsqueeze(0)])
        return torch.stack(RFMemory)   

    def merge_mem_attn_dynamic(self, frames, frames_len):
        mem_embeddings = []
        long_memory_buffer = []
        for frame_i, frame in enumerate(frames):
            long_memory_buffer.append(frame)
            if frame_i + 1 in frames_len:
                tokenizer = getattr(self, "_tokenizer", None)
                qms_ratio = getattr(self, "_qms_ratio", 0.5)
                if tokenizer is not None and len(long_memory_buffer) >= 1:
                    out = self.mem_select(list(long_memory_buffer), frame_i, qms_ratio=qms_ratio, tokenizer=tokenizer)
                    mem_embeddings.append(out)
                else:
                    mem_embeddings.append(torch.stack(long_memory_buffer))
            if len(long_memory_buffer) >= 2:
                similar = F.cosine_similarity(long_memory_buffer[-1], long_memory_buffer[-2]).mean()
                if similar > self.eps:
                    scores = self.visual_attention_model(long_memory_buffer[-2], long_memory_buffer[-1])
                    new_frame_feature = self.attention_merge(long_memory_buffer[-2], long_memory_buffer[-1], scores)
                    long_memory_buffer[-2] = new_frame_feature
                    del(long_memory_buffer[-1])
                elif len(long_memory_buffer) >= 3:
                    similar = self.visual_attention_model(long_memory_buffer[-1], torch.cat(long_memory_buffer[:-1]), type="attn")
                    if similar > self.eps:
                        multi_scores = self.visual_attention_model(torch.cat(long_memory_buffer[:-1]), long_memory_buffer[-1])
                        merge_memory = self.attention_merge(torch.cat(long_memory_buffer[:-1]), long_memory_buffer[-1], multi_scores)
                        long_memory_buffer = list(torch.split(merge_memory, [len(i) for i in long_memory_buffer[:-1]], dim=0))
        return mem_embeddings

    def visual_embed(self, frames: torch.Tensor, frames_lens, user_frames=None, user_qs=None, qms_ratio=0.5, tokenizer=None):
        self._tokenizer = tokenizer
        self._qms_ratio = qms_ratio
        if hasattr(self, 'vision_encode'):
            if frames.dim() >= 3 and frames.shape[-1] == self.config.vision_hidden_size:
                pass
            else:
                with torch.cuda.amp.autocast():
                    frames = self.vision_encode(self.vision_encoder, frames)
                frames = frames.to(self.dtype)
        mem_frames = []
        visual_memory_ = []
        n = len(frames_lens)
        user_frames = user_frames if user_frames is not None else [[]] * n
        user_qs = user_qs if user_qs is not None else [[]] * n
        for idx, (frame, frame_len) in enumerate(zip(frames, frames_lens)):
            self.user_frames = list(user_frames[idx]) if idx < len(user_frames) else []
            self.user_qs = list(user_qs[idx]) if idx < len(user_qs) else []
            self.history_q = None
            mem_embeddings = self.merge_mem_attn(frame, frame_len)
            visual_memory_.append([len(i) for i in mem_embeddings])
            mem_frames.append(torch.cat(mem_embeddings))
        embed_frames = torch.stack(mem_frames)
        embed_frames = self.connector(embed_frames)
        embed_frames = embed_frames.view(-1, embed_frames.shape[-1])
        return embed_frames, visual_memory_

    def build_from_plan(self, visual_memory_ = None, add_generation_prompt=False, **kwargs):
        plan = kwargs["plans"][0]
        
        visual_memory_s = visual_memory_[0]
        visual_memory_s.reverse()
        
        system_prompt = kwargs["system_prompt"]
        tokenizer = kwargs["tokenizer"]
        
        oft = "<v>" * self.config.frame_num_tokens
        
        text_parts = []
        offset = 0
        learn_ranges = []
        normal_attn_mask = []
        always_attn_mask = []
        cancel_attn_mask = []

        conversation = [{"role": "system", "content": system_prompt}]
        text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=add_generation_prompt)
        text_parts.append(text)
        offset += len(text)

        normal_attn_mask.append(range(0, offset))
        always_attn_mask.append(range(0, offset))

        for item in plan:
            role = item["role"]
            if role == "stream":
                for i in item.get("sampled_frames", []):
                    visual_memory_len = visual_memory_s.pop() - 1
                    memory_text = "Memory: [{}]\nCurrent: [{}]\n".format(",".join([oft for _ in range(visual_memory_len)]), oft)
                    continue_text = tokenizer.eos_token + "\n"
                    total_text = memory_text + continue_text
                    text_parts.append(total_text)
                    continue_len = len(continue_text)
                    normal_attn_mask.append(range(offset, offset + len(total_text)))
                    cancel_attn_mask.append(range(offset + len(total_text) - continue_len, offset + len(total_text)))
                    learn_ranges.append(range(offset + len(memory_text), offset + len(total_text) - 1))

                    offset += len(total_text)

            elif role == "user":
                memory_text = ""
                user_text = "User: {} ({} s)\n".format(item["original"]["content"], item["original"]["fps_time"])
                if item.get("memory") is not None:
                    visual_memory_len = visual_memory_s.pop() - 1
                    memory_text = "Memory: [{}]\nCurrent: [{}]\n".format(",".join([oft for _ in range(visual_memory_len)]), oft)
                    memory_len = len(memory_text)
                    total_text = memory_text + user_text
                    text_parts.append(total_text)

                    normal_attn_mask.append(range(offset, offset + len(total_text)))
                    always_attn_mask.append(range(offset + memory_len, offset + len(total_text)))
                else:
                    total_text = user_text
                    text_parts.append(total_text)

                    normal_attn_mask[-1] = range(normal_attn_mask[-1].start, offset + len(total_text))
                    always_attn_mask.append(range(offset, offset + len(total_text)))

                offset += len(total_text)

            elif role == "assistant":
                
                assistant_text = "Assistant: {}{}".format(item["original"]["content"], tokenizer.eos_token)
                time_text = " ({} s)\n".format(item["original"]["fps_time"])
                memory_text = ""
                memory_len = 0

                if item.get("memory") is not None:
                    visual_memory_len = visual_memory_s.pop() - 1
                    memory_text = "Memory: [{}]\nCurrent: [{}]\n".format(",".join([oft for _ in range(visual_memory_len)]), oft)
                    memory_len = len(memory_text)

                total_text = memory_text + assistant_text + time_text
                text_parts.append(total_text)
                if item.get("memory") is not None:
                    normal_attn_mask.append(range(offset, offset + len(total_text)))
                    always_attn_mask.append(range(offset + memory_len, offset + len(total_text)))
                    learn_ranges.append(range(offset + memory_len, offset + memory_len + len(assistant_text)))
                else:
                    normal_attn_mask[-1] = range(normal_attn_mask[-1].start, offset + len(total_text))
                    always_attn_mask[-1] = range(always_attn_mask[-1].start, offset + len(total_text))
                    learn_ranges.append(range(offset, offset + len(assistant_text)))
                        
                offset += len(total_text)

        return "".join(text_parts), learn_ranges, (normal_attn_mask, always_attn_mask, cancel_attn_mask)

    def get_input(self, text, learn_ranges, masks, **kwargs):
        tokenizer = kwargs["tokenizer"]
        batch = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt", padding=True)
        batch_labels = torch.full_like(batch.input_ids, LabelSmoother.ignore_index, dtype=torch.long)
        
        attention_masks = torch.zeros((batch.input_ids.shape[0], batch.input_ids.shape[1], batch.input_ids.shape[1]), dtype=torch.long)
        position_ids = torch.zeros_like(batch.input_ids, dtype=torch.long)
        offset = 0
        
        for mask_id, (labels, input_ids, offset_mapping, learn_range, mask) in enumerate(zip(
            batch_labels, batch.input_ids, batch.offset_mapping, [learn_ranges], [masks]
        )):
            max_len = len(input_ids)
            
            for learn_r in learn_range:
                start, stop = self.search(offset_mapping, learn_r, max_len)
                labels[start - 1:stop - 1] = input_ids[start:stop]

            
            for normal_attn_mask in mask[0]:
                start, stop = self.search(offset_mapping, normal_attn_mask, max_len)
                attn_tril = torch.tril(torch.ones(stop - start, stop - start), diagonal=0)
                attention_masks[mask_id][start:stop, start:stop] = attn_tril
                position_ids[mask_id][start:stop] = torch.linspace(0, stop - start - 1, stop - start)

            for always_attn_mask in mask[1]:
                start, stop = self.search(offset_mapping, always_attn_mask, max_len)
                attention_masks[mask_id][stop:, start:stop] += 1
                position_ids[mask_id][stop:] += position_ids[mask_id][stop - 1] + 1 - offset
                offset = position_ids[mask_id][stop - 1] + 1

            for cancel_attn_mask in mask[2]:
                start, stop = self.search(offset_mapping, cancel_attn_mask, max_len)
                attention_masks[mask_id][stop:, start:stop] *= 0
                if stop != max_len and tokenizer.decode(input_ids[stop]) != 'Memory':
                    position_ids[mask_id][stop:] -= stop - start

            labels[labels >= len(tokenizer) - 1] = tokenizer.eos_token_id
        
        return batch["input_ids"].to(self.model.device), batch_labels.to(self.model.device), attention_masks.bool().to(self.model.device), position_ids.to(self.model.device)
            
    def search(self, mapping, ranging, max_len):
        start = torch.nonzero(mapping[:,0] == ranging.start).squeeze()
        if mapping[:,0][-1] >= ranging.stop:
            stop = torch.nonzero(mapping[:,0] == ranging.stop).squeeze()
        else:
            stop = max_len
        return start, stop

    def joint_embed(
        self,
        frames: torch.Tensor = None,
        frames_lens: torch.Tensor = None,
        **kwargs
    ):
        assert frames is not None
        visual_embeds, visual_memory_ = self.visual_embed(
            frames, frames_lens,
            user_frames=kwargs.get("user_frames"),
            user_qs=kwargs.get("user_qs"),
            qms_ratio=kwargs.get("qms_ratio", 0.5),
            tokenizer=kwargs.get("tokenizer"),
        )
        text, learn_ranges, attn_masks = self.build_from_plan(visual_memory_, **kwargs)
        input_ids, labels, attention_masks, position_ids = self.get_input(text, learn_ranges, attn_masks, **kwargs)
        inputs_embeds = self.get_input_embeddings()(input_ids.clamp(max=self.vocab_size-1))
        v_mask = input_ids == self.config.v_placeholder_id
        if v_mask.any():
            inputs_embeds[v_mask] = visual_embeds
        return input_ids, inputs_embeds, labels, attention_masks, position_ids


    def attention(self, turing_memory, new_feature, update_ratio=None):
        if update_ratio is None:
            update_ratio = getattr(self, "attention_update_ratio", 0.2)
        T1, D1 = turing_memory.shape
        T2, D2 = new_feature.shape
        assert D1 == D2, f"dimension not match, {D1} != {D2}"
        weight = self.attention_model(turing_memory, new_feature)
        weight = weight * update_ratio
        decay = weight.sum(dim=1, keepdim=True)
        turing_memory = turing_memory * (1 - decay) + torch.mm(weight, new_feature)
        return turing_memory

    def attention_mem(self, img_feature, video_max_frames, update_ratio=0.2):
        T, P, D = img_feature.shape
        T0 = video_max_frames
        if T <= T0:
            return img_feature
        cur_feature = img_feature[:T0]
        turing_memory = cur_feature.reshape(T0*P, D)
        for i in range(T0, T, T0):
            j = min(i + T0, T)
            new_feature = img_feature[i:j]
            new_feature = new_feature.reshape(-1, D)
            turing_memory = self.attention(turing_memory, new_feature, update_ratio=update_ratio)
        cur_feature = turing_memory.reshape(T0, P, D)
        return cur_feature


    @torch.no_grad()
    def stream_evaluate(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        frames: torch.ByteTensor,
        ignore_token_id: int = -100,
        frame_token_interval_threshold: float = 0.0,
        **kwargs
    ):
        assert input_ids.size(0) == labels.size(0) == 1
        input_id, label = input_ids[0], labels[0]
        device = input_id.device
        zero = torch.tensor(0, dtype=torch.int, device=device)
        one = torch.tensor(1, dtype=torch.int, device=device)
        turn_stops = ((input_id == self.config.eos_token_id).nonzero() + 1)[:,0].tolist()
        turn_starts = [0] + turn_stops[:-1]
        num_turns = len(turn_starts)
        outputs = self.forward(input_ids=input_ids, frames=frames, return_dict=True, use_cache=True)
        logit, past_key_values = outputs.logits[0], outputs.past_key_values
        v_placeholder_id = self.config.v_placeholder_id
        use_interval = self.config.frame_token_interval_id is not None
        frame_token_interval_id = self.config.frame_token_interval_id if use_interval else self.config.eos_token_id
        frame_num_tokens = self.config.frame_token_cls
        if self.config.frame_token_pooled:
            frame_num_tokens += self.config.frame_token_pooled[0] * self.config.frame_token_pooled[1]
        past_num_frames = 0
        lm_ppls, frame_diffs, fluencies, lm_correctness = [], [], [], []
        for r, (turn_start, turn_stop) in enumerate(zip(turn_starts, turn_stops)):
            turn_label = label[turn_start:turn_stop]
            turn_learn_mask = turn_label != ignore_token_id
            if not turn_learn_mask.any():
                continue
            turn_logit = logit[turn_start:turn_stop]
            turn_input_id = input_id[turn_start:turn_stop]
            turn_v_mask = turn_input_id == v_placeholder_id
            turn_num_frames = turn_v_mask.sum() // frame_num_tokens
            turn_stream_mask = turn_v_mask & turn_learn_mask
            turn_lm_mask = turn_learn_mask & ~turn_stream_mask

            if turn_lm_mask.any():
                turn_lm_masked_logit, turn_lm_masked_label = turn_logit[turn_lm_mask], turn_label[turn_lm_mask]
                lm_ppl = torch.nn.functional.cross_entropy(turn_lm_masked_logit, turn_lm_masked_label).exp()
                lm_ppls.append(lm_ppl)
                turn_lm_masked_wrong_mask = turn_lm_masked_logit.argmax(dim=-1) != turn_lm_masked_label
                if turn_lm_masked_wrong_mask.any():
                    num_lm_correct_tokens = turn_lm_masked_wrong_mask.nonzero()[0,0]
                else:
                    num_lm_correct_tokens = (~turn_lm_masked_wrong_mask).sum()
                lm_correctness.append(num_lm_correct_tokens / turn_lm_masked_label.numel())

            if turn_stream_mask.any():
                turn_score = turn_logit.softmax(dim=-1)
                turn_stream_masked_score = turn_score[turn_stream_mask]
                if frame_token_interval_threshold > 0:
                    lower_threshold_mask = turn_stream_masked_score[:, frame_token_interval_id] < frame_token_interval_threshold
                    turn_stream_masked_score[lower_threshold_mask] = 0
                turn_stream_masked_pred_mask = turn_stream_masked_score.argmax(dim=-1) != frame_token_interval_id
                if turn_stream_masked_pred_mask.any():
                    frame_diff = turn_stream_mask.sum() - turn_stream_masked_pred_mask.nonzero()[0,0] - 1
                else:
                    turn_last_stream_idx = turn_stream_mask.nonzero()[-1,0]
                    past_key_values_before_assistant = self.trim_past_key_values(past_key_values, 0, turn_start + turn_last_stream_idx + 1)
                    if r == num_turns - 1:
                        frame_diff = zero
                    else:
                        next_turn_num_frames = (input_id[turn_starts[r+1]:turn_stops[r+1]] == v_placeholder_id).sum() // frame_num_tokens
                        to_append_num_frames = min(next_turn_num_frames, turn_num_frames - 1)
                        if to_append_num_frames == 0:
                            frame_diff = zero
                        else:
                            to_append_frames = frames[past_num_frames+turn_num_frames:past_num_frames+turn_num_frames+to_append_num_frames]
                            frame_placeholder = [v_placeholder_id] * frame_num_tokens
                            if use_interval:
                                frame_placeholder = [frame_token_interval_id] + frame_placeholder
                            to_append_input_id = torch.tensor(frame_placeholder * to_append_num_frames, dtype=torch.long, device=device)
                            to_append_logit = self.forward(
                                input_ids=to_append_input_id[None],
                                past_key_values=past_key_values_before_assistant,
                                frames=to_append_frames,
                                return_dict=True, use_cache=True
                            ).logits[0]
                            idxs = torch.arange(len(frame_placeholder)-1, len(to_append_input_id), len(frame_placeholder), device=device)
                            to_append_score = to_append_logit[idxs].softmax(dim=-1)
                            if frame_token_interval_threshold > 0:
                                lower_threshold_mask = to_append_score[:, frame_token_interval_id] < frame_token_interval_threshold
                                to_append_score[lower_threshold_mask] = 0
                            to_append_score_pred_mask = to_append_score.argmax(dim=-1) != frame_token_interval_id
                            if to_append_score_pred_mask.any():
                                frame_diff = -(to_append_score_pred_mask.nonzero()[0,0] + 1)
                            else:
                                frame_diff = -to_append_num_frames
                frame_diffs.append(frame_diff.abs())
            if turn_lm_mask.any() and turn_stream_mask.any():
                num_learn_v_tokens = turn_stream_mask.sum()
                num_learn_valid_tokens = turn_lm_masked_label.numel() + num_learn_v_tokens
                if frame_diff == 0:
                    fluency = (num_learn_v_tokens + num_lm_correct_tokens) / num_learn_valid_tokens
                elif frame_diff > 0:
                    fluency = (num_learn_v_tokens - frame_diff) / num_learn_valid_tokens
                else:
                    fluency = (num_learn_v_tokens - 1) / num_learn_valid_tokens
                fluencies.append(fluency)
            past_num_frames += turn_num_frames
        lm_ppl = torch.stack(lm_ppls).mean() if lm_ppls else one
        frame_diff = torch.stack(frame_diffs).float().mean() if frame_diffs else zero
        fluency = torch.stack(fluencies).float().mean() if fluencies else one
        lm_correctness = torch.stack(lm_correctness).float().mean() if lm_correctness else one
        return torch.stack([lm_ppl, frame_diff, fluency, lm_correctness])

    def trim_past_key_values(self, past_key_values, start, stop):
        return [[past_keys[:,:,start:stop], past_values[:,:,start:stop]] for past_keys, past_values in past_key_values]

def fast_greedy_generate(*, model: LiveMixin, inputs_embeds: torch.Tensor, past_key_values: Cache, eos_token_id: int, inplace_output_ids: torch.Tensor):
    for i in range(inplace_output_ids.size(1)):
        outputs = model(inputs_embeds=inputs_embeds, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        new_token_id = outputs.logits[:, -1:].argmax(dim=-1)
        inplace_output_ids[:, i] = new_token_id
        if new_token_id == eos_token_id:
            break
        inputs_embeds = model.get_input_embeddings()(new_token_id)
    return inplace_output_ids[:, :i+1], past_key_values

def fast_greedy_generate_mem(*, model: LiveMixin, inputs_embeds: torch.Tensor, past_key_values: Cache, eos_token_id: int, inplace_output_ids: torch.Tensor, position_ids: torch.Tensor, video_time_ids: torch.Tensor):
    for i in range(inplace_output_ids.size(1)):
        outputs = model(inputs_embeds=inputs_embeds, past_key_values=past_key_values, use_cache=True, position_ids=position_ids)
        past_key_values = outputs.past_key_values
        position_ids = position_ids[:, -1:] + 1
        new_token_id = outputs.logits[:, -1:].argmax(dim=-1)
        inplace_output_ids[:, i] = new_token_id
        if new_token_id == eos_token_id:
            break
        inputs_embeds = model.get_input_embeddings()(new_token_id)
    inputs_embeds = model.get_input_embeddings()(torch.cat([new_token_id, video_time_ids], dim=-1))
    _, lens, _ = inputs_embeds.shape
    position_ids = position_ids + torch.linspace(0, lens - 1, lens).to(inputs_embeds.device).unsqueeze(0)
    outputs = model(inputs_embeds=inputs_embeds, past_key_values=past_key_values, use_cache=True, position_ids=position_ids)
    past_key_values = outputs.past_key_values
    position_ids = position_ids[:, -1:] + 1
    return inplace_output_ids[:, :i+1], past_key_values, position_ids[0][0].int().item()

def build_live(
    *,
    is_training: bool,
    config_class: type,
    model_class: type,
    llm_pretrained: str = None,
    finetune_modules: list[str] = None,
    lora_modules: str = None,
    lora_r: int = None,
    lora_alpha: int = None,
    set_vision_inside: bool = False,
    resume_from_checkpoint: str = '',
    attn_implementation: str = 'flash_attention_2',
    torch_dtype: str | torch.dtype = 'auto',
    **kwargs
):
    if is_training:
        config = config_class.from_pretrained(llm_pretrained, **kwargs)
        config.mem_length = kwargs['mem_length']
        config.cutoff_len = kwargs['cutoff_len']
        model = model_class.from_pretrained(llm_pretrained, config=config, torch_dtype=torch_dtype, attn_implementation=attn_implementation)
    else:   
        config = config_class.from_pretrained(llm_pretrained, **kwargs)
        config.mem_length = kwargs['mem_length']
        config.cutoff_len = kwargs['cutoff_len']
        model = model_class.from_pretrained(llm_pretrained, config=config, torch_dtype=torch_dtype, attn_implementation=attn_implementation)

    tokenizer = build_live_tokenizer_and_update_config(llm_pretrained, model.config)
    model.set_connector_inside()
    model.set_attention_inside()

    model.eps = kwargs['eps']
    model.attention_update_ratio = kwargs.get('attention_update_ratio', 0.2)
    model.merge_mem_attn = model.merge_mem_attn_dynamic

    if is_training:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_modules,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            modules_to_save=finetune_modules,
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        if resume_from_checkpoint:
            model = PeftModel.from_pretrained(model, resume_from_checkpoint, is_trainable=False, torch_device='cpu')
        else:
            logger.warning(f'!!! Fail to load checkpoint: {resume_from_checkpoint}. Return a new initialized model.')
        if set_vision_inside:
            model.set_vision_inside()
        model.requires_grad_(False)
    return model, tokenizer
