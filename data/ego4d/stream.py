import torch
import random
from transformers import PreTrainedTokenizer

from models.tokenization_live import chat_template_mem_offsets


class StreamMemoryIn(torch.utils.data.Dataset):
    def __init__(
        self,
        is_training: bool,
        system_prompt: str,
        augmentation: bool,
        max_num_frames: int,
        tokenizer: PreTrainedTokenizer,
        **kwargs
    ):
        super().__init__()
        self.is_training = is_training
        self.system_prompt = system_prompt
        self.system_prompt_mem = kwargs["system_prompt_mem"]
        self.augmentation = augmentation
        self.tokenizer = tokenizer
        self.max_num_frames = max_num_frames
        self.chat_template_offsets = chat_template_mem_offsets(tokenizer)
        self.mem_length = kwargs["mem_length"]
        self.frame_num_tokens = kwargs["frame_num_tokens"]
        self.cutoff_len = kwargs["cutoff_len"]
        self.assistant_target_ratio = kwargs.get("assistant_target_ratio", 1.0)
        self.plan_sample_extra_tokens = kwargs.get("plan_sample_extra_tokens", 20)
        self.continue_sample_p_cap = kwargs.get("continue_sample_p_cap", 10.0)
        assert system_prompt is not None, "Please add a system prompt"

    def __getitem__(self, *, anno, **kwargs):
        plan, frames_num, user_frames, user_qs = self.collect_conversation_plan(anno, assistant_target_ratio=self.assistant_target_ratio)
        load_path, load_ranges = [(k, v) for k, v in anno["load_ranges"].items()][0]
        frames = torch.load(load_path, map_location="cpu", weights_only=True)
        if isinstance(frames, dict):
            frames = next(iter(frames.values()))
        frames = frames[load_ranges]
        return plan, frames, frames_num, user_frames, user_qs

    def collect_conversation_plan(self, anno, assistant_target_ratio=0.5):
        plan = []
        start_frames = 0
        frames_num = []
        user_frames = []
        user_qs = []
        all_continue_frames = []
        reversed_frames = set()
        extra = self.plan_sample_extra_tokens
        cap = self.continue_sample_p_cap
        assistant_sample_p = (self.cutoff_len / 2 - anno["tokens"]) / (self.mem_length * self.frame_num_tokens + extra) / anno["stream_num"]
        continue_sample_p = min((self.cutoff_len / 4 - anno["tokens"]) / (self.mem_length * self.frame_num_tokens + extra) / anno["stream_num"], cap)
        sample_num = max(1, int(continue_sample_p))

        continue_num = 0
        assistant_num = 0
        for c_id, message in enumerate(anno["conversation"]):
            try:
                task_weight = self.task_weights[message["task"]]
            except Exception:
                task_weight = 1
            role = message["role"]
            item = {"role": role, "original": message, "start_frames": start_frames}

            if role == "stream":
                num_frames = message["num_frames"]
                item["num_frames"] = num_frames
                item["sampled_frames"] = []

                if random.uniform(0, 1) <= continue_sample_p and num_frames > 1:
                    if c_id == len(anno["conversation"]) - 1 or anno["conversation"][c_id + 1]["role"] == "user":
                        pick = random.sample(range(start_frames + 1, start_frames + num_frames + 1), min(sample_num, num_frames))
                    else:
                        pick = random.sample(range(start_frames + 1, start_frames + num_frames), min(sample_num, num_frames - 1))
                    item["sampled_frames"] = sorted(pick)
                    continue_num += len(item["sampled_frames"])
                    frames_num.extend(item["sampled_frames"])
                    all_continue_frames.extend(item["sampled_frames"])

                start_frames += num_frames

            elif role == "user":
                user_frames.append(start_frames)
                user_qs.append(item["original"]["content"])
                if start_frames not in frames_num:
                    frames_num.append(start_frames)
                    item["memory"] = start_frames
                else:
                    item["memory"] = None
                reversed_frames.add(start_frames)

            elif role == "assistant":
                item["learn"] = False
                if start_frames in frames_num:
                    item["memory"] = None
                    reversed_frames.add(start_frames)
                elif random.uniform(0, 1) <= assistant_sample_p and random.uniform(0, 1) <= task_weight:
                    frames_num.append(start_frames)
                    item["memory"] = start_frames
                    assistant_num += 1
                    reversed_frames.add(start_frames)
                else:
                    continue

            plan.append(item)

        allowed_continue = int(assistant_num * assistant_target_ratio)
        current_continue = continue_num
        prunable_continue_frames = [f for f in all_continue_frames if f not in reversed_frames]
        if current_continue > allowed_continue:
            delete_frames = random.sample(
                prunable_continue_frames,
                min(current_continue - allowed_continue, len(prunable_continue_frames)),
            )
            keep_frames = set(all_continue_frames) - set(delete_frames)
            for item in plan:
                if item["role"] == "stream":
                    item["sampled_frames"] = [frame for frame in item["sampled_frames"] if frame in keep_frames]
            frames_num = sorted(list(set(keep_frames) | set(reversed_frames)))
        return plan, frames_num, user_frames, user_qs
