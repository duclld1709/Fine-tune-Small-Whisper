import argparse
import glob
import itertools
import math
import os
from functools import partial

from datasets import load_dataset_builder
import torch
from torch.utils.data import DataLoader, IterableDataset
import wandb

from src.dataset.vimd_loader import collate_fn, load_vimd_hf_dataset, load_vivos_hf_dataset
from src.features.mel_extractor import LogMelExtractor
from src.models.squeezeformer import SqueezeformerEncoder
from src.tokenizer.train_tokenizer import Tokenizer
from src.training.trainer import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Squeezeformer on ViMD/VIVOS datasets")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["vimd_val", "vimd_test", "vivos_test"],
        choices=["vimd_val", "vimd_test", "vivos_test"],
        help="Evaluation targets to run",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path (default: latest checkpoint)")
    parser.add_argument("--tokenizer-model", type=str, default="vimd_unigram.model")
    parser.add_argument("--vocab-size", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit per target split")
    parser.add_argument("--no-streaming", action="store_true", help="Disable HF streaming mode")
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu, or auto if omitted")

    parser.add_argument("--vimd-dataset", type=str, default="nguyendv02/ViMD_Dataset")
    parser.add_argument("--vimd-config", type=str, default=None)
    parser.add_argument("--vivos-dataset", type=str, default="AILAB-VNUHCM/vivos")
    parser.add_argument("--vivos-config", type=str, default=None)

    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="vimd-asr-squeezeformer")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


def find_latest_checkpoint():
    checkpoints = glob.glob("squeezeformer_epoch_*.pt")
    if not checkpoints:
        raise FileNotFoundError("No checkpoints found matching squeezeformer_epoch_*.pt")
    checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return checkpoints[-1]


def load_checkpoint_state_dict(checkpoint_path, device):
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)

    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            clean_state_dict[k[10:]] = v
        else:
            clean_state_dict[k] = v
    return clean_state_dict


def maybe_limit_dataset(dataset, max_samples):
    if max_samples is None or max_samples <= 0:
        return dataset

    if hasattr(dataset, "take"):
        return dataset.take(max_samples)

    if isinstance(dataset, IterableDataset):
        return _LimitedIterableDataset(dataset, max_samples)

    if hasattr(dataset, "select"):
        return dataset.select(range(min(max_samples, len(dataset))))

    return dataset


class _LimitedIterableDataset(IterableDataset):
    def __init__(self, base_dataset, max_samples):
        super().__init__()
        self.base_dataset = base_dataset
        self.max_samples = max_samples

    def __iter__(self):
        return itertools.islice(iter(self.base_dataset), self.max_samples)

    def __len__(self):
        if hasattr(self.base_dataset, "__len__"):
            try:
                return min(len(self.base_dataset), self.max_samples)
            except TypeError:
                pass
        return self.max_samples


def build_eval_targets(args, streaming):
    return {
        "vimd_val": lambda: load_vimd_hf_dataset(
            split="valid",
            dataset_name=args.vimd_dataset,
            config_name=args.vimd_config,
            streaming=streaming,
        ),
        "vimd_test": lambda: load_vimd_hf_dataset(
            split="test",
            dataset_name=args.vimd_dataset,
            config_name=args.vimd_config,
            streaming=streaming,
        ),
        "vivos_test": lambda: load_vivos_hf_dataset(
            split="test",
            dataset_name=args.vivos_dataset,
            config_name=args.vivos_config,
            streaming=streaming,
        ),
    }


def estimate_split_num_examples(dataset_name, split, config_name=None):
    """
    Try to fetch split size from HF dataset metadata so tqdm can estimate ETA.
    """
    try:
        builder = load_dataset_builder(dataset_name, name=config_name)
    except Exception:
        return None

    split_infos = getattr(builder.info, "splits", None) or {}
    if split in split_infos and split_infos[split].num_examples is not None:
        return int(split_infos[split].num_examples)

    alias_map = {
        "valid": ["valid", "validation", "val", "dev"],
        "val": ["val", "valid", "validation", "dev"],
        "validation": ["validation", "valid", "val", "dev"],
        "test": ["test", "eval", "evaluation"],
        "train": ["train", "training"],
    }
    for candidate in alias_map.get(split, []):
        if candidate in split_infos and split_infos[candidate].num_examples is not None:
            return int(split_infos[candidate].num_examples)
    return None


def get_target_dataset_spec(args, target_name):
    if target_name == "vimd_val":
        return args.vimd_dataset, "valid", args.vimd_config
    if target_name == "vimd_test":
        return args.vimd_dataset, "test", args.vimd_config
    if target_name == "vivos_test":
        return args.vivos_dataset, "test", args.vivos_config
    raise ValueError(f"Unknown target: {target_name}")


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    streaming = not args.no_streaming

    checkpoint_path = args.checkpoint if args.checkpoint is not None else find_latest_checkpoint()
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.exists(args.tokenizer_model):
        raise FileNotFoundError(f"Tokenizer model not found: {args.tokenizer_model}")

    print(f"Using device: {device}")
    print(f"Loading checkpoint: {checkpoint_path}")
    tokenizer = Tokenizer(args.tokenizer_model)

    model = SqueezeformerEncoder(
        input_dim=80,
        num_layers=16,
        d_model=256,
        num_heads=4,
        ffn_dim=1024,
        conv_kernel=31,
        dropout=0.0,
        vocab_size=args.vocab_size,
    ).to(device)
    model.load_state_dict(load_checkpoint_state_dict(checkpoint_path, device))
    model.eval()

    feature_extractor = LogMelExtractor(
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        win_length=400,
        n_mels=80,
        f_min=0,
        f_max=8000,
    ).to(device)

    wandb_run = None
    if not args.no_wandb:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "checkpoint": checkpoint_path,
                "targets": args.targets,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "streaming": streaming,
                "max_samples": args.max_samples,
                "vimd_dataset": args.vimd_dataset,
                "vivos_dataset": args.vivos_dataset,
            },
        )

    targets = build_eval_targets(args, streaming=streaming)
    collate = partial(collate_fn, tokenizer=tokenizer)
    results = {}

    for target_name in args.targets:
        print(f"\n=== Evaluating {target_name} ===")
        dataset = targets[target_name]()
        dataset = maybe_limit_dataset(dataset, args.max_samples)
        dataset_name, target_split, target_config = get_target_dataset_spec(args, target_name)

        split_num_examples = estimate_split_num_examples(
            dataset_name=dataset_name,
            split=target_split,
            config_name=target_config,
        )
        if args.max_samples is not None and args.max_samples > 0:
            if split_num_examples is None:
                total_examples = args.max_samples
            else:
                total_examples = min(split_num_examples, args.max_samples)
        else:
            total_examples = split_num_examples

        if total_examples is None:
            try:
                total_examples = len(dataset)
            except TypeError:
                total_examples = None

        total_batches = math.ceil(total_examples / args.batch_size) if total_examples is not None else None
        if total_batches is None:
            print(f"{target_name}: running with unknown total size (ETA unavailable).")
        else:
            print(
                f"{target_name}: ~{total_examples} samples ({total_batches} batches) "
                "for progress/ETA."
            )

        loader_kwargs = {
            "batch_size": args.batch_size,
            "collate_fn": collate,
            "num_workers": args.num_workers,
            "pin_memory": device.type == "cuda",
        }
        if args.num_workers > 0:
            loader_kwargs["persistent_workers"] = True

        dataloader = DataLoader(dataset, **loader_kwargs)
        metrics = evaluate(
            model=model,
            dataloader=dataloader,
            device=device,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            beam_search_decoder=None,
            return_details=True,
            progress_desc=f"Evaluating {target_name}",
            progress_total=total_batches,
        )
        results[target_name] = metrics

        print(
            f"{target_name}: "
            f"WER={metrics['wer']:.4f}, "
            f"CER={metrics['cer']:.4f}, "
            f"Inference={metrics['inference_seconds']:.2f}s, "
            f"RTF={metrics['rtf']:.4f}"
        )

        if wandb_run is not None:
            wandb.log(
                {
                    f"{target_name}/wer": metrics["wer"],
                    f"{target_name}/cer": metrics["cer"],
                    f"{target_name}/inference_seconds": metrics["inference_seconds"],
                    f"{target_name}/audio_seconds": metrics["audio_seconds"],
                    f"{target_name}/rtf": metrics["rtf"],
                    f"{target_name}/samples_per_second": metrics["samples_per_second"],
                    f"{target_name}/num_samples": metrics["num_samples"],
                }
            )

    print("\n=== Summary ===")
    for target_name, metrics in results.items():
        print(
            f"{target_name}: "
            f"WER={metrics['wer']:.4f}, CER={metrics['cer']:.4f}, "
            f"Inference={metrics['inference_seconds']:.2f}s, "
            f"Samples={metrics['num_samples']}"
        )

    if wandb_run is not None:
        for target_name, metrics in results.items():
            for key, value in metrics.items():
                wandb.run.summary[f"{target_name}/{key}"] = value
        wandb.finish()


if __name__ == "__main__":
    main()
