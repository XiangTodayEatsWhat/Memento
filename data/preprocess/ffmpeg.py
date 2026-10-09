"""
Sample video frames: resample to fixed FPS and resolution (e.g. 2 FPS, max 384).
Output: video_dir_2fps_384 (or video_dir_{fps}fps_max{resolution}).

Run from Memento root:
  PYTHONPATH=. python -m data.preprocess.ffmpeg --video_dir datasets/videos --frame_fps 2 --frame_resolution 384
  PYTHONPATH=. python -m data.preprocess.ffmpeg --video_dir datasets/videos --frame_fps 2 --frame_resolution 384 --num_workers 8
"""
import os
import sys
import argparse
from pathlib import Path
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from data.utils import ffmpeg_once


def _task(args):
    src_path, dst_path, fps, resolution, pad = args
    if os.path.exists(dst_path):
        return 0
    try:
        ffmpeg_once(src_path, dst_path, fps=fps, resolution=resolution, pad=pad)
        return 0
    except Exception as e:
        print(f"Error {src_path}: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory of input videos (.mp4)")
    parser.add_argument("--frame_fps", type=int, default=2,
                        help="Output frame rate (default 2)")
    parser.add_argument("--frame_resolution", type=int, default=384,
                        help="Max resolution (default 384)")
    parser.add_argument("--pad", type=str, default="#000000")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Parallel workers (default 1)")
    args = parser.parse_args()

    src_root = args.video_dir.rstrip("/")
    dst_root = src_root
    if args.frame_fps is not None:
        dst_root += f"_{args.frame_fps}fps"
    if args.frame_resolution is not None:
        dst_root += f"_max{args.frame_resolution}"

    pather = Path(src_root)
    src_paths = [str(p) for p in pather.rglob("*") if p.is_file() and str(p).endswith(".mp4")]
    tasks = []
    for src_path in src_paths:
        dst_path = src_path.replace(src_root, dst_root)
        tasks.append((src_path, dst_path, args.frame_fps, args.frame_resolution, args.pad))

    print(f"Resampling {len(tasks)} videos: {src_root} -> {dst_root}")
    if args.num_workers <= 1:
        err = 0
        for t in tasks:
            err += _task(t)
    else:
        with Pool(args.num_workers) as pool:
            err = sum(pool.map(_task, tasks))
    print(f"Done. Errors: {err}")


if __name__ == "__main__":
    main()
