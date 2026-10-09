# Getting started

Run all commands from the repository root.

## 1. Environment

Use Linux, Python 3.11, NVIDIA GPUs with BF16 support, and a driver compatible with PyTorch 2.10. The reference recipe uses four 80 GB GPUs. Feature encoding and inference can run on one GPU. Required memory depends on video length, dialogue history and generation length.

Install FFmpeg and development tools through your system package manager; both ffmpeg and ffprobe must be on PATH.

    git clone https://github.com/XiangTodayEatsWhat/Memento.git
    cd Memento
    python3.11 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    DS_BUILD_OPS=0 python -m pip install -r requirements.txt
    python -m pip install -r requirements-publish.txt -r requirements-eval.txt
    python -m unittest discover -s tests -v

DeepSpeed may compile required operations at runtime. Install a compatible CUDA toolkit and C++ build toolchain if JIT compilation is needed. Use the pinned dependencies for compatibility with the custom model and trainer.

## 2. Download model and annotation assets

Obtain access to Llama 3.1 from its model provider and authenticate with Hugging Face when needed. Authentication uses your Hugging Face login or `HF_TOKEN`.

    hf auth login
    python scripts/download_assets.py --kind base --output checkpoints/Llama-3.1-8B-Instruct
    python scripts/download_assets.py --kind vision --output checkpoints/siglip-large-patch16-384
    python scripts/download_assets.py --kind dataset --output datasets/Memento-54K

For an existing trained adapter, also run:

    python scripts/download_assets.py --kind model --output checkpoints/Memento-8B

Use --revision with a Hub commit hash to pin a particular release. Training from scratch does not require the Memento adapter download.

The dataset includes 53,626 training examples from 4,426 videos and 198 test examples from 40 disjoint videos. All examples contain the video UID and timestamped dialogue.

    python datasets/Memento-54K/uids/verify_dataset.py --train datasets/Memento-54K/data/train.jsonl --test datasets/Memento-54K/data/test.jsonl

## 3. Download source videos

Obtain your own Ego4D access through the [official download procedure](https://ego4d-data.org/docs/start-here/) and configure the supplied AWS credentials. Source videos are downloaded from Ego4D, not redistributed in this repository or the annotation dataset.

    python -m pip install ego4d
    python scripts/download_videos.py --uid-dir datasets/Memento-54K/uids --split all --output-dir datasets/ego4d --aws-profile ego4d

The default is Ego4D v2 video_540ss. The expected raw-video directory is datasets/ego4d/v2/video_540ss. To download only evaluation videos, use --split test. 

## 4. Prepare visual features

Run preprocessing for the annotations you intend to use:

    python scripts/prepare_features.py --annotations datasets/Memento-54K/data/train.jsonl datasets/Memento-54K/data/test.jsonl --videos datasets/ego4d/v2/video_540ss --work-dir datasets/prepared --vision checkpoints/siglip-large-patch16-384 --gpu 0 --batch-size 8

The pipeline resamples to 2 FPS, scales the long side to 384 pixels, pads to 384×384, and extracts one global plus 3×3 spatial SigLIP tokens per frame. Each output is a BF16 tensor shaped [frames, 10, 1024].

The exact output path is recorded in datasets/prepared/prepared.json. Use it in subsequent commands:

    FEATURES=$(python -c 'import json; print(json.load(open("datasets/prepared/prepared.json"))["feature_dir"])')

Use a separate work directory for each video selection. Preprocessing can resume existing outputs, but source content and model version must remain unchanged; use a fresh work directory after changing either. Video decoding currently loads each resampled video into host memory, so long videos require adequate RAM.

For existing cached features, validate them and rebuild metadata instead:

    python scripts/validate_features.py --annotations datasets/Memento-54K/data/train.jsonl datasets/Memento-54K/data/test.jsonl --features "$FEATURES"

## 5. Train

    python scripts/train.py --config configs/original_checkpoint.json --annotations datasets/Memento-54K/data/train.jsonl --features "$FEATURES" --output outputs/memento --llm checkpoints/Llama-3.1-8B-Instruct --vision checkpoints/siglip-large-patch16-384 --gpus 0,1,2,3

The released checkpoint recipe in `configs/original_checkpoint.json` uses one epoch, AdamW, learning rate 1e-4, cosine decay, LoRA rank 128/alpha 256 and a 32,768-token cutoff. Per-device batch size is 1 with two accumulation steps (effective batch size 8 on four GPUs), warmup ratio 0.05 and seed 42. Dynamic Memory uses threshold 0.7 and update ratio 0.2; QMS ratio is 0.5.

Use --config to provide another JSON recipe and --port to avoid distributed-port conflicts. An existing checkpoint in the output directory is resumed by the training entry point; use a new output directory for a new experiment. The final directory contains adapter_model.safetensors, adapter_config.json, tokenizer files and a training completion marker.


## 6. Streaming inference and temporal evaluation

    python scripts/evaluate.py --annotations datasets/Memento-54K/data/test.jsonl --features "$FEATURES" --checkpoint outputs/memento --output outputs/evaluation --llm checkpoints/Llama-3.1-8B-Instruct --vision checkpoints/siglip-large-patch16-384 --gpus 0,1,2,3

Use --gpus 0 for single-GPU evaluation. A downloaded model can be used by setting --checkpoint checkpoints/Memento-8B. Every evaluation example processes its entire video. Evaluation resumes matching completed results.

Outputs:

    outputs/evaluation/infer_results/            One complete JSON result per example
    outputs/evaluation/metrics/temporal_metrics.json
    outputs/evaluation/metrics/merged_predictions.json
    outputs/evaluation/metrics/judge_requests_review.jsonl

Temporal evaluation checks complete sample coverage and reports TimeRecall and Redundancy. No external API is called by this command.

## 7. Answer-quality evaluation

The paper's quality judge is gpt-3.5-turbo-0125. Configure OPENAI_API_KEY outside the repository. OPENAI_BASE_URL can select a compatible authorized provider; it must serve the same model for a comparable result.

    python benchmark/judge.py --metrics outputs/evaluation/metrics

This sends questions, reference answers and model responses to the configured API and incurs API charges. Completed requests are cached; `--aggregate-only` recomputes metrics from saved responses.

See [evaluation conventions](evaluation.md) for time windows, long-duration categories and quality-score denominators.

## 8. Export a trained adapter

    python scripts/export_model.py --checkpoint outputs/memento --output outputs/Memento-8B

The exported Safetensors include LoRA weights and the learned connector and attention modules. Base Llama and SigLIP weights are obtained separately. The export replaces machine-specific base-model paths with the public model identifier and writes model configuration and weight checksums. Use the Memento inference entry point; a generic text-generation pipeline does not implement streaming memory.

## License

Code: [Apache-2.0](../LICENSE). Memento annotation text: CC BY 4.0, as described in the dataset card. Model weights are subject to the Llama 3.1 Community License and Acceptable Use Policy. Ego4D source videos remain subject to the licenses granted by their providers. These are separate assets with separate terms; see [third-party notices](../THIRD_PARTY_NOTICES.md).
