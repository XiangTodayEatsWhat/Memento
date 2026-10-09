from dataclasses import dataclass, field
from transformers import TrainingArguments
import os


@dataclass
class LiveMemoryTrainingArguments(TrainingArguments):
    system_prompt: str = (
        "A multimodal AI assistant is helping users with some activities."
        " Below is their conversation, interleaved with the list of video frames received by the assistant."
    )
    train_datasets: list[str] = None
    eval_datasets: list[str] = None
    stream_loss_weight: float = 1.0
    llm_pretrained: str = field(default_factory=lambda: os.environ.get('MEMORIAL_LLM', 'meta-llama/Llama-3.1-8B-Instruct'))
    vision_pretrained: str = field(default_factory=lambda: os.environ.get('MEMORIAL_VISION', 'google/siglip-large-patch16-384'))
    lora_modules: str = "model.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|lm_head$"
    lora_r: int = 128
    lora_alpha: int = 256
    finetune_modules: list[str] = field(default_factory=lambda: ['connector', 'attention_model'])
    frame_fps: int = 2
    frame_token_cls: bool = True
    frame_resolution: int = 384
    frame_token_interval: str = '........'
    frame_token_interval_threshold: float = 0.0
    frame_token_pooled: list[int] = field(default_factory=lambda: [3, 3])
    augmentation: bool = False
    attn_implementation: str = 'sdpa'
    output_dir: str = 'outputs/debug'
    infer_output_path: str = 'none.json'

    mem_length: int = 16
    cutoff_len: int = 16384
    eps: float = 0.7
    pt_size: int = 3
    qms_ratio: float = 0.5

    waiting_frames_max: int = 20
    same_time_user_delta: float = 0.5
    fallback_tokens: int = 500
    task_weight_scale: float = 5.0
    task_weight_min: float = 0.1
    assistant_target_ratio: float = 1.0
    plan_sample_extra_tokens: int = 20
    continue_sample_p_cap: float = 10.0
    attention_update_ratio: float = 0.2

    embed_path: str = field(default_factory=lambda: os.environ.get('MEMORIAL_EMBED', ''))

    max_num_frames: int = -1

    frame_num_tokens: int = None
    embed_mark: str = None

    system_prompt_mem: str = "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n"

    def __post_init__(self):
        super().__post_init__()
        if self.frame_num_tokens is None:
            self.frame_num_tokens = self.pt_size ** 2 + 1
        if self.embed_mark is None:
            self.embed_mark = '2fps_384_1+{}x{}'.format(self.pt_size, self.pt_size)
