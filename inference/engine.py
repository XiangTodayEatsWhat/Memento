"""
Memento inference engine: Dynamic Memory only.
Run from repo root: PYTHONPATH=. python -m inference.engine  (or import LiveMemInfer)
"""
import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torchvision
import transformers
import collections
from dataclasses import asdict
from torchvision.io import read_video

from models import build_model_and_tokenizer, parse_args, fast_greedy_generate_mem

logger = transformers.logging.get_logger('liveinfer')
import torch.nn.functional as F

# Dynamic memory only
MEM_TYPE = "dynamic"


class LiveMemInfer:
    def __init__(self, device=None, infer_output_path=None, args=None, set_vision_inside=False):
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if isinstance(device, int):
            device = torch.device(f'cuda:{device}')
        self.device = device
        args = args if args is not None else parse_args()
        if infer_output_path is not None:
            args.infer_output_path = infer_output_path
        self.model, self.tokenizer = build_model_and_tokenizer(
            is_training=False, set_vision_inside=set_vision_inside, **asdict(args)
        )
        self.model.to(self.device)
        self.model.eval()

        self.hidden_size = self.model.config.hidden_size
        self.frame_fps = args.frame_fps
        self.frame_interval = 1 / self.frame_fps
        self.frame_resolution = self.model.config.frame_resolution
        self.frame_num_tokens = self.model.config.frame_num_tokens
        self.frame_placeholder_ids = torch.tensor(self.model.config.v_placeholder_id)
        self.answer_id = self.tokenizer.encode("Assistant", add_special_tokens=False)[0]
        self.video_max_frames = args.mem_length
        self.oft = "<v>" * self.model.config.frame_num_tokens
        self.eps = self.model.eps
        self.qms_ratio = args.qms_ratio

        self.system_prompt_mem = args.system_prompt
        self.inplace_output_ids = torch.zeros(1, 100, device=self.device, dtype=torch.long)
        self.eos_token_id = self.model.config.eos_token_id
        conversation = [{"role": "system", "content": self.system_prompt_mem}]
        self._start_ids = self.tokenizer.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=False
        )
        self._start_ids = torch.tensor(self._start_ids).to(self.device)
        self._added_stream_prompt_ids = self.tokenizer.apply_chat_template(
            [{}], add_stream_prompt=True, return_tensors='pt'
        ).to(self.device)
        self.infer_output_path = getattr(args, 'infer_output_path', 'outputs/infer')
        self.reset()

    def update_past_key_values(self, past, pre_offset, post_offset):
        past_key_values = []
        for i in past:
            k = torch.cat([i[0][:, :, :pre_offset], i[0][:, :, post_offset:]], dim=2)
            v = torch.cat([i[1][:, :, :pre_offset], i[1][:, :, post_offset:]], dim=2)
            past_key_values.append((k, v))
        self.past_key_values = tuple(past_key_values)

    def _call_for_response(self, inputs_embeds, video_time, query, update=True):
        _, memory_offset, _ = inputs_embeds.shape
        if query is not None:
            user_text = "User: {} ({} s)\n".format(query, video_time)
            user_ids = torch.tensor(
                self.tokenizer.encode(user_text, add_special_tokens=False)
            ).to(self.device)
            user_embeds = self.model.get_input_embeddings()(
                user_ids.clamp(max=self.tokenizer.vocab_size - 1)
            )
            inputs_embeds = torch.cat([
                inputs_embeds,
                user_embeds.view(1, -1, self.hidden_size)
            ], dim=1)
        batch, lens, channel = inputs_embeds.shape
        self.position_ids = torch.linspace(0, lens - 1, lens).to(self.device).unsqueeze(0) + self.offset
        _, _, pre_offset, _ = self.past_key_values[0][0].shape
        post_offset = pre_offset + memory_offset
        time_text = " ({} s)\n".format(video_time)
        time_ids = torch.tensor(
            self.tokenizer.encode(time_text, add_special_tokens=False)
        ).to(self.device).unsqueeze(0)
        if update:
            output_ids, tmp_past_key_values, self.offset = fast_greedy_generate_mem(
                model=self.model,
                inputs_embeds=inputs_embeds,
                past_key_values=self.past_key_values,
                eos_token_id=self.eos_token_id,
                inplace_output_ids=self.inplace_output_ids,
                position_ids=self.position_ids,
                video_time_ids=time_ids
            )
            self.update_past_key_values(tmp_past_key_values, pre_offset=pre_offset, post_offset=post_offset)
        else:
            output_ids, tmp_past_key_values, _ = fast_greedy_generate_mem(
                model=self.model,
                inputs_embeds=inputs_embeds,
                past_key_values=self.past_key_values,
                eos_token_id=self.eos_token_id,
                inplace_output_ids=self.inplace_output_ids,
                position_ids=self.position_ids,
                video_time_ids=time_ids
            )
        response = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return query, response

    def _call_for_streaming(self):
        while self.frame_embeds_queue:
            video_time, frame_embeds = self.frame_embeds_queue.popleft()
            memory_text = "Memory: [{}]\nCurrent: [{}]\n".format(
                ",".join([self.oft] * (len(frame_embeds) - 1)), self.oft
            )
            memory_ids = torch.tensor(
                self.tokenizer.encode(memory_text, add_special_tokens=False)
            ).to(self.device)
            inputs_embeds = self.model.get_input_embeddings()(
                memory_ids.clamp(max=self.tokenizer.vocab_size - 1)
            )
            v_mask = memory_ids == self.frame_placeholder_ids
            if v_mask.any():
                inputs_embeds[v_mask] = frame_embeds.flatten(0, 1)
            inputs_embeds = inputs_embeds.view(1, -1, self.hidden_size)
            if not self.past_key_values:
                inputs_embeds = torch.cat([
                    self.model.get_input_embeddings()(self._start_ids).view(1, -1, self.hidden_size),
                    inputs_embeds,
                ], dim=1)
            batch, lens, channel = inputs_embeds.shape
            self.position_ids = torch.linspace(0, lens - 1, lens).to(self.device).unsqueeze(0) + self.offset
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=self.past_key_values,
                position_ids=self.position_ids
            )
            if self.offset == 0:
                self.offset += len(self._start_ids)
                self.update_past_key_values(
                    outputs.past_key_values, pre_offset=self.offset, post_offset=lens
                )
                inputs_embeds = inputs_embeds[:, self.offset:]
            if self.query_queue and self.video_time >= self.query_queue[0][0]:
                video_time, query = self.query_queue.popleft()
                return inputs_embeds, video_time, query
            next_score = outputs.logits[:, -1:].softmax(dim=-1)
            if next_score[:, :, self.answer_id] < self.frame_token_interval_threshold:
                next_score[:, :, self.answer_id].zero_()
            last_id = next_score.argmax(dim=-1)
            if last_id == self.answer_id:
                return inputs_embeds, video_time, None
        return None, None, None

    def mem_select(self, RFMemory, video_time):
        if len(self.query_queue) > 0 and video_time >= self.query_queue[0][0]:
            user_q = self.query_queue[0][1] + "\n"
            user_ids = self.tokenizer(
                user_q, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.model.device)
            user_embeds = self.model.get_input_embeddings()(
                user_ids.clamp(max=self.model.vocab_size - 1)
            )
            self.history_q = user_embeds if self.history_q is None else torch.cat(
                [self.history_q, user_embeds], dim=1
            )
        if self.history_q is not None and len(RFMemory) >= 2:
            Memory = torch.stack(RFMemory[:-1])
            corr = self.model.multi_attention_model(Memory, self.history_q)
            topk_indices = torch.topk(corr, k=max(int(self.qms_ratio * len(corr)), 1)).indices
            Memory = Memory[topk_indices] * corr[topk_indices].unsqueeze(-1).unsqueeze(-1)
            return torch.cat([Memory, RFMemory[-1].unsqueeze(0)])
        return torch.stack(RFMemory)

    def merge_mem_attn_dynamic(self, frame, video_time):
        self.long_memory_buffer.append(frame)
        if self.qms_ratio != 1.0:
            self.frame_embeds_queue.append((
                video_time,
                self.model.connector(self.mem_select(self.long_memory_buffer, video_time))
            ))
        else:
            self.frame_embeds_queue.append((
                video_time,
                self.model.connector(torch.stack(self.long_memory_buffer))
            ))
        if len(self.long_memory_buffer) >= 2:
            similar = F.cosine_similarity(
                self.long_memory_buffer[-1], self.long_memory_buffer[-2]
            ).mean()
            if similar > self.eps:
                scores = self.model.visual_attention_model(
                    self.long_memory_buffer[-2], self.long_memory_buffer[-1]
                )
                new_frame_feature = self.model.attention_merge(
                    self.long_memory_buffer[-2], self.long_memory_buffer[-1], scores
                )
                self.long_memory_buffer[-2] = new_frame_feature
                del self.long_memory_buffer[-1]
            elif len(self.long_memory_buffer) >= 3:
                cat_memory = torch.cat(self.long_memory_buffer[:-1])
                similar = self.model.visual_attention_model(
                    self.long_memory_buffer[-1], cat_memory, type="attn"
                )
                if similar > self.eps:
                    multi_scores = self.model.visual_attention_model(
                        cat_memory, self.long_memory_buffer[-1]
                    )
                    merge_memory = self.model.attention_merge(
                        cat_memory, self.long_memory_buffer[-1], multi_scores
                    )
                    self.long_memory_buffer = list(torch.split(
                        merge_memory,
                        [i.size(0) for i in self.long_memory_buffer[:-1]],
                        dim=0
                    ))
        if torch.cuda.is_available():
            pass  # Keep reusable allocator blocks rather than flush every frame

    def reset(self):
        self.query_queue = collections.deque()
        self.frame_embeds_queue = collections.deque()
        self.video_time = 0
        self.last_frame_idx = -1
        self.video_tensor = None
        self.embed_tensor = None
        self.last_ids = torch.tensor([[]], device=self.device, dtype=torch.long)
        self.past_key_values = None
        self.offset = 0
        self.similar_list = []
        self.long_memory_buffer = []
        self.frame_token_interval_threshold = 0.5
        self.history_q = None

    def input_query_stream(self, query, video_time=None):
        if video_time is None:
            self.query_queue.append((self.video_time, query))
        else:
            self.query_queue.append((video_time, query))
        if not self.past_key_values:
            return '(NOTE: No video stream. Please upload or select a video first.)'
        return '(NOTE: Question queued. Answer will appear as the stream processes.)'

    def input_video_stream(self, video_time):
        """Feed video stream (dynamic memory only)."""
        frame_idx = int(video_time * self.frame_fps)
        if frame_idx > self.last_frame_idx:
            ranger = range(self.last_frame_idx + 1, frame_idx + 1)
            with torch.cuda.amp.autocast():
                frames = self.model.vision_encode(
                    self.model.vision_encoder, self.video_tensor[ranger]
                )
            frames = frames.to(self.model.dtype)
            if frames.dim() == 3:
                for i, r in enumerate(ranger):
                    self.merge_mem_attn_dynamic(frames[i], r / self.frame_fps)
            else:
                for r, fe in zip(ranger, frames.split(self.frame_num_tokens)):
                    self.merge_mem_attn_dynamic(fe, r / self.frame_fps)
        self.last_frame_idx = frame_idx
        self.video_time = video_time

    def input_embed_stream(self, video_time):
        """Feed precomputed embeddings (e.g. for benchmark). Dynamic memory only."""
        frame_idx = int(video_time * self.frame_fps)
        if frame_idx > self.last_frame_idx:
            ranger = range(self.last_frame_idx + 1, frame_idx + 1)
            frames_embeds = self.embed_tensor[ranger].flatten(0, 1).split(self.frame_num_tokens)
            for r, frame_embeds in zip(ranger, frames_embeds):
                self.merge_mem_attn_dynamic(frame_embeds, r / self.frame_fps)
        self.last_frame_idx = frame_idx
        self.video_time = video_time

    def load_embed(self, embed_path, q_start, q_end):
        self.start_frame = int(q_start * self.frame_fps)
        self.end_frame = int(q_end * self.frame_fps)
        self.embed_tensor = torch.load(embed_path, map_location=self.device)[
            self.start_frame : self.end_frame + 1
        ].to(self.device)
        self.num_video_frames = self.embed_tensor.size(0)
        self.video_duration = self.embed_tensor.size(0) / self.frame_fps
        logger.warning(f'{embed_path} -> {self.embed_tensor.shape}, {self.frame_fps} FPS')

    def load_video(self, video_path):
        self.video_tensor = read_video(
            video_path, pts_unit='sec', output_format='TCHW'
        )[0].to(self.device)
        self.num_video_frames = self.video_tensor.size(0)
        self.video_duration = self.video_tensor.size(0) / self.frame_fps
        logger.warning(f'{video_path} -> {self.video_tensor.shape}, {self.frame_fps} FPS')

    def __call__(self, update=True):
        if not self.frame_embeds_queue:
            return None, None
        input_embeds, video_time, query = self._call_for_streaming()
        response = None
        if video_time is not None:
            query, response = self._call_for_response(
                input_embeds, video_time, query, update=update
            )
        return query, response
