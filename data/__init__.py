from torch.utils.data import ConcatDataset
from functools import partial

from .data_collator import get_data_collator

try:
    from .ego4d import build_ego4d_memento_train
except ImportError:
    build_ego4d_memento_train = None

__all__ = [
    "build_concat_train_dataset",
    "build_eval_dataset_dict",
    "get_data_collator",
    "get_compute_metrics_dict",
]


def _build_memento_from_path(anno_path: str, is_training: bool, **kwargs):
    from .ego4d.memento import Ego4DMemento
    kwargs = dict(kwargs)
    embed_path = kwargs.pop("embed_path", None) or ""
    if not embed_path:
        raise ValueError("embed_path (or MEMORIAL_EMBED) is required for training from JSONL.")
    return Ego4DMemento(anno_path=anno_path, embed_path=embed_path, is_training=is_training, **kwargs)


def build_concat_train_dataset(train_datasets=None, is_training=True, **kwargs):
    if train_datasets is None or len(train_datasets) == 0:
        return None
    datasets = []
    for name_or_path in train_datasets:
        if isinstance(name_or_path, str) and (name_or_path.endswith(".jsonl") or name_or_path.endswith(".json")):
            datasets.append(_build_memento_from_path(name_or_path, is_training=is_training, **kwargs))
        else:
            name = name_or_path if isinstance(name_or_path, str) else getattr(name_or_path, "name", None)
            builder = globals().get(f"build_{name}") if name else None
            if builder is None and build_ego4d_memento_train is not None and name == "ego4d_memento_train":
                builder = build_ego4d_memento_train
            if builder is None:
                raise ValueError(f"Unknown dataset: {name_or_path}. Use a JSONL path or register build_{name}.")
            datasets.append(builder(is_training=is_training, **kwargs))
    return ConcatDataset(datasets)


def build_eval_dataset_dict(eval_datasets=None, is_training=False, **kwargs):
    if eval_datasets is None or len(eval_datasets) == 0:
        return None
    out = {}
    for name_or_path in eval_datasets:
        if isinstance(name_or_path, str) and (name_or_path.endswith(".jsonl") or name_or_path.endswith(".json")):
            key = name_or_path.split("/")[-1].replace(".jsonl", "").replace(".json", "")
            out[key] = _build_memento_from_path(name_or_path, is_training=is_training, **kwargs)
        else:
            name = name_or_path if isinstance(name_or_path, str) else getattr(name_or_path, "name", None)
            builder = globals().get(f"build_{name}") if name else None
            if builder is None and build_ego4d_memento_train is not None and name == "ego4d_memento_train":
                builder = build_ego4d_memento_train
            if builder is None:
                raise ValueError(f"Unknown eval dataset: {name_or_path}")
            out[name_or_path] = builder(is_training=is_training, **kwargs)
    return out


def get_compute_metrics_dict(dataset_dict=None, **kwargs):
    if not dataset_dict:
        return None
    return {k: getattr(v, "compute_metrics", lambda **kw: {}) for k, v in dataset_dict.items()}
