import json
import os
import random
import tqdm
from collections import defaultdict

from .ego4d import Ego4D
from .stream import StreamMemoryIn
from ..utils import ceil_time_by_fps, floor_time_by_fps, DictWithTo


class Ego4DMemento(Ego4D, StreamMemoryIn):
    evaluation_kwargs = DictWithTo(evaluator="generate")

    def __init__(
        self,
        *,
        anno_path: str,
        embed_path: str,
        frame_fps: int,
        is_training: bool,
        tokenizer=None,
        **kwargs
    ):
        Ego4D.__init__(self, embed_dir=embed_path, frame_fps=frame_fps, is_training=is_training, tokenizer=tokenizer, **kwargs)
        StreamMemoryIn.__init__(
            self,
            frame_fps=frame_fps,
            is_training=is_training,
            tokenizer=tokenizer,
            **kwargs
        )
        self.anno_path = anno_path
        self.is_training = is_training
        self.frame_fps = frame_fps
        self.task_distributions = defaultdict(int)
        self.annos = []

        with open(anno_path) as f:
            lines = f.readlines()
        for anno_id, line in enumerate(tqdm.tqdm(lines, desc="load jsonl")):
            line = line.strip()
            if not line:
                continue
            anno = json.loads(line)
            video_uid = anno.get("video_uid")
            if video_uid not in self.metadata:
                continue
            duration = self.metadata[video_uid]["duration"]
            if not anno.get("conversation"):
                continue
            conv0 = anno["conversation"][0]
            role = conv0.get("role")
            time = float(conv0.get("time", 0))
            content = conv0.get("content", "")
            task = conv0.get("task", "unknown")
            if not (time >= 0 and time <= duration and content):
                continue

            self.task_distributions[task] += 1
            fps_time = floor_time_by_fps(time, frame_fps, 0, duration)
            waiting_max = kwargs.get("waiting_frames_max", 20)
            waiting_frames = random.randint(1, min(waiting_max, int(fps_time * frame_fps) + 1))
            conversation = []
            if waiting_frames:
                conversation.append({"role": "stream", "num_frames": waiting_frames, "learn": waiting_frames - 1})
            start_fps_time = fps_time - (waiting_frames - 1) / frame_fps
            conversation.append({"role": "user", "content": content, "time": time, "fps_time": fps_time - start_fps_time})

            for message in anno["conversation"][1:]:
                role = message.get("role")
                content = message.get("content", "")
                time = float(message.get("time", 0))
                task = message.get("task", "unknown")
                self.task_distributions[task] += 1
                if time > duration:
                    break
                if time < conversation[-1]["time"]:
                    time = conversation[-1]["time"]
                if time == conversation[-1]["time"]:
                    if role == "user":
                        time += kwargs.get("same_time_user_delta", 0.5)
                    else:
                        if conversation[-1]["role"] == "user":
                            conversation.append({"role": "assistant", "content": content, "time": time, "fps_time": conversation[-1]["fps_time"], "learn": True, "task": task})
                        else:
                            conversation[-1]["content"] += "\n" + content
                            conversation[-1]["task"] += "\n" + task
                        continue
                if role == "user":
                    fps_time = floor_time_by_fps(time, frame_fps, conversation[-1]["fps_time"] + start_fps_time, duration)
                    if fps_time > duration:
                        break
                    fps_time -= start_fps_time
                    if fps_time > conversation[-1]["fps_time"]:
                        conversation.append({"role": "stream", "num_frames": int((fps_time - conversation[-1]["fps_time"]) * frame_fps), "learn": True})
                    conversation.append({"role": "user", "content": content, "time": time, "fps_time": fps_time})
                else:
                    fps_time = ceil_time_by_fps(time, frame_fps, conversation[-1]["fps_time"] + start_fps_time, duration)
                    if fps_time > duration:
                        break
                    fps_time -= start_fps_time
                    if fps_time > conversation[-1]["fps_time"]:
                        conversation.append({"role": "stream", "num_frames": int((fps_time - conversation[-1]["fps_time"]) * frame_fps), "learn": True})
                    conversation.append({"role": "assistant", "content": content, "time": time, "fps_time": fps_time, "learn": True, "task": task})

            if not conversation:
                continue
            for conv_i, conv in enumerate(conversation):
                if "fps_time" in conv:
                    conversation[conv_i]["str_time"] = str(conv["fps_time"])

            path = self.metadata[video_uid]["path"]
            end_frame = int((conversation[-1]["fps_time"] + start_fps_time) * frame_fps) + 1
            start_frame = int(start_fps_time * frame_fps)
            self.annos.append({
                "tokens": len(tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=False)) if tokenizer else kwargs.get("fallback_tokens", 500),
                "stream_num": sum(1 for i in conversation if i["role"] == "stream"),
                "anno_id": anno_id,
                "conversation": conversation,
                "load_ranges": {path: range(start_frame, end_frame)},
            })

        min_count = min(self.task_distributions.values()) if self.task_distributions else 1
        scale = kwargs.get("task_weight_scale", 5.0)
        min_w = kwargs.get("task_weight_min", 0.1)
        self.task_weights = {
            cls: round(max(min_count / count * scale, min_w), 6)
            for cls, count in self.task_distributions.items()
        }

    def __getitem__(self, index):
        anno = self.annos[index]
        plan, frames, frames_num, user_frames, user_qs = StreamMemoryIn.__getitem__(self, anno=anno)
        return plan, frames, frames_num, user_frames, user_qs, index, self.evaluation_kwargs

    def __len__(self):
        return len(self.annos)


def build_ego4d_memento_train(**kwargs):
    return Ego4DMemento(**kwargs)
