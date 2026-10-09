import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NCCL_P2P_LEVEL", "NVL")

from dataclasses import asdict
from transformers import set_seed

from models import build_model_and_tokenizer, parse_args
from data import build_concat_train_dataset, build_eval_dataset_dict, get_data_collator, get_compute_metrics_dict
from engine import TrainerWithGenToEval


def check_scd_folder(path):
    if not os.path.exists(path):
        return False
    for folder in os.listdir(path):
        if folder.startswith('checkpoint') and os.path.isdir(os.path.join(path, folder)):
            return True
    return False


def train():
    args = parse_args()
    set_seed(args.seed)

    model, tokenizer = build_model_and_tokenizer(is_training=True, **asdict(args))
    train_dataset = build_concat_train_dataset(tokenizer=tokenizer, **asdict(args))
    if train_dataset is None:
        raise ValueError("train_dataset is None. Configure train_datasets and implement build_concat_train_dataset in data/__init__.py.")
    eval_dataset_dict = build_eval_dataset_dict(tokenizer=tokenizer, **asdict(args))

    if not eval_dataset_dict:
        args.eval_strategy = "no"
        args.eval_on_start = False

    data_collator = get_data_collator(tokenizer=tokenizer, **asdict(args))
    compute_metrics_dict = get_compute_metrics_dict(
        dataset_dict=eval_dataset_dict, tokenizer=tokenizer, **asdict(args)
    )

    args.gradient_checkpointing_kwargs = {'use_reentrant': False}
    args.resume_from_checkpoint = check_scd_folder(args.output_dir)
    trainer = TrainerWithGenToEval(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset_dict,
        data_collator=data_collator,
        compute_metrics=compute_metrics_dict,
    )

    trainer.train(args.resume_from_checkpoint)
    trainer.save_model()
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if eval_dataset_dict:
        metrics = {}
        for eval_name, eval_ds in eval_dataset_dict.items():
            trainer.compute_metrics = compute_metrics_dict[eval_name]
            metrics.update(
                trainer.evaluate(
                    eval_dataset=eval_ds,
                    metric_key_prefix=f"eval_{eval_name}",
                )
            )
        print(metrics)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    train()
