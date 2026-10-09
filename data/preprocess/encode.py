"""
Extract vision embeddings from resampled videos (output of ffmpeg step).
Uses the same vision encoder and token layout as training (e.g. 1+3x3, SigLIP).
Output: video_dir_2fps_384_1+3x3_google--siglip-large-patch16-384 (one .pt per video).

Run from Memento root:
  PYTHONPATH=. python -m data.preprocess.encode --video_dir datasets/videos_2fps_384 --vision_pretrained google/siglip-large-patch16-384
"""
import os
import sys
import argparse
from dataclasses import asdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from models.configuration_live import LiveConfigMixin
from models.arguments_live import LiveMemoryTrainingArguments
from models.vision_live import build_live_vision
from data.utils import distributed_encode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory of resampled videos (output of ffmpeg step, e.g. ..._2fps_384)")
    parser.add_argument("--vision_pretrained", type=str, default="google/siglip-large-patch16-384")
    parser.add_argument("--pt_size", type=int, default=3,
                        help="Spatial token size (default 3 -> 1+3x3 tokens per frame)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--save_bf16", action="store_true", help="Save embeddings in bfloat16")
    args = parser.parse_args()

    # Same config as training (Live1Mem) for consistent embed_mark
    train_args = LiveMemoryTrainingArguments(
        vision_pretrained=args.vision_pretrained,
        pt_size=args.pt_size,
    )
    embed_mark = train_args.embed_mark  # e.g. 2fps_384_1+3x3
    vision_config = LiveConfigMixin(**asdict(train_args))
    _, vision_encode = build_live_vision(vision_config)

    distributed_encode(
        src_root=args.video_dir,
        vision_pretrained=args.vision_pretrained,
        vision_encode=vision_encode,
        batch_size=args.batch_size,
        embed_mark=embed_mark,
        save_bf16=args.save_bf16,
    )
    dst_root = f"{args.video_dir.rstrip('/')}_{embed_mark}_{args.vision_pretrained.replace('/', '--')}"
    print(f"Embeddings saved to {dst_root}")


if __name__ == "__main__":
    main()
