import os
import argparse
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial
import math

from src.dataset.vimd_loader import (
    load_vimd_dataset,
    load_vivos_hf_dataset,
    load_vivos_transcripts,
    collate_fn,
)
from src.tokenizer.train_tokenizer import (
    train_sentencepiece_tokenizer,
    train_sentencepiece_tokenizer_from_texts,
    Tokenizer,
)
from src.features.mel_extractor import LogMelExtractor, SpecAugment
from src.models.squeezeformer import SqueezeformerEncoder
from src.training.trainer import train_epoch, evaluate
from src.decoding.ctc_beam_search import ctc_beam_search

import wandb

DEFAULT_ESTIMATED_SAMPLES = {
    "vimd": 15021,
    "vivos": 11660,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train Squeezeformer on ViMD or VIVOS")
    parser.add_argument(
        "--train-dataset",
        type=str,
        default="vimd",
        choices=["vimd", "vivos"],
        help="Select training dataset pipeline.",
    )
    parser.add_argument(
        "--vimd-data-files",
        type=str,
        default="Code/src/data/train/*.parquet",
        help="Parquet glob for ViMD training data.",
    )
    parser.add_argument(
        "--vivos-dataset",
        type=str,
        default="AILAB-VNUHCM/vivos",
        help="HF dataset id for VIVOS.",
    )
    parser.add_argument("--vivos-config", type=str, default=None)
    parser.add_argument("--vivos-split", type=str, default="train")
    parser.add_argument(
        "--model-prefix",
        type=str,
        default=None,
        help="Tokenizer prefix; defaults to vimd_unigram or vivos_unigram by train-dataset.",
    )
    parser.add_argument(
        "--estimated-samples",
        type=int,
        default=None,
        help="Override dataset size estimate for scheduler when streaming datasets do not expose len().",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for this run.")
    parser.add_argument("--batch-size", type=int, default=16, help="Effective batch size.")
    parser.add_argument("--gradient-accumulation", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate.")
    parser.add_argument("--vocab-size", type=int, default=2000, help="Tokenizer/model vocabulary size.")
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="Initialize model weights from an existing checkpoint for continued training.",
    )
    parser.add_argument(
        "--tokenizer-model",
        type=str,
        default=None,
        help="Use an existing tokenizer model file and skip tokenizer training.",
    )
    parser.add_argument("--no-streaming", action="store_true", help="Disable HF streaming mode.")
    return parser.parse_args()


def _safe_len(dataset):
    try:
        return len(dataset)
    except Exception:
        return None


def _build_training_dataset(args, streaming):
    if args.train_dataset == "vimd":
        return load_vimd_dataset(data_files=args.vimd_data_files, streaming=streaming)

    return load_vivos_hf_dataset(
        split=args.vivos_split,
        dataset_name=args.vivos_dataset,
        config_name=args.vivos_config,
        streaming=streaming,
    )


def _train_tokenizer_if_needed(args, vocab_size, model_prefix):
    if os.path.exists(f"{model_prefix}.model"):
        return

    print("Training Tokenizer...")
    if args.train_dataset == "vimd":
        train_sentencepiece_tokenizer(
            data_files=args.vimd_data_files,
            vocab_size=vocab_size,
            model_prefix=model_prefix,
        )
        return

    vivos_texts = load_vivos_transcripts(
        split=args.vivos_split,
        dataset_name=args.vivos_dataset,
    )
    train_sentencepiece_tokenizer_from_texts(
        vivos_texts,
        vocab_size=vocab_size,
        model_prefix=model_prefix,
    )


def _load_checkpoint_state_dict(checkpoint_path, device):
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)

    clean_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            clean_state_dict[key[10:]] = value
        else:
            clean_state_dict[key] = value
    return clean_state_dict


def _infer_epoch_offset(checkpoint_path):
    if checkpoint_path is None:
        return 0
    match = re.search(r"epoch_(\d+)\.pt$", os.path.basename(checkpoint_path))
    if match is None:
        return 0
    return int(match.group(1))


def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    streaming = not args.no_streaming
    model_prefix = args.model_prefix or f"{args.train_dataset}_unigram"
    
    # Initialize wandb
    wandb.init(project="vimd-asr-squeezeformer", config={
        "train_dataset": args.train_dataset,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "vocab_size": args.vocab_size,
        "model_prefix": model_prefix,
        "streaming": streaming,
        "vimd_data_files": args.vimd_data_files,
        "vivos_dataset": args.vivos_dataset,
        "vivos_config": args.vivos_config,
        "vivos_split": args.vivos_split,
        "estimated_samples": args.estimated_samples,
        "init_checkpoint": args.init_checkpoint,
        "tokenizer_model": args.tokenizer_model,
    })
    
    # Configuration
    vocab_size = wandb.config.vocab_size
    model_prefix = wandb.config.model_prefix
    
    batch_size = wandb.config.batch_size
    gradient_accumulation = wandb.config.gradient_accumulation
    per_device_batch_size = batch_size // gradient_accumulation
    if per_device_batch_size < 1:
        raise ValueError(
            f"Invalid batch setup: batch_size={batch_size}, "
            f"gradient_accumulation={gradient_accumulation}. "
            "Ensure batch_size >= gradient_accumulation."
        )
    epochs = wandb.config.epochs
    learning_rate = wandb.config.learning_rate
    weight_decay = 1e-4
    
    # 1. Tokenizer setup
    if args.tokenizer_model:
        if not os.path.exists(args.tokenizer_model):
            raise FileNotFoundError(f"Tokenizer model not found: {args.tokenizer_model}")
        tokenizer = Tokenizer(args.tokenizer_model)
        print(f"Using existing tokenizer: {args.tokenizer_model}")
    else:
        _train_tokenizer_if_needed(args=args, vocab_size=vocab_size, model_prefix=model_prefix)
        tokenizer = Tokenizer(f"{model_prefix}.model")

    vocab_size = tokenizer.vocab_size
    wandb.config.update({"effective_vocab_size": vocab_size}, allow_val_change=True)
    
    # 2. Dataset and DataLoader
    dataset = _build_training_dataset(args=args, streaming=streaming)
    dataset_size = _safe_len(dataset)
    if dataset_size is not None:
        print(f"Detected {dataset_size} samples for {args.train_dataset}.")
    else:
        print(f"Dataset length unavailable for {args.train_dataset} (streaming mode).")
    
    feature_extractor = LogMelExtractor(
        sample_rate=16000, n_fft=400, hop_length=160, win_length=400, n_mels=80, f_min=0, f_max=8000
    ).to(device)
    
    collate = partial(collate_fn, tokenizer=tokenizer)
    
    # using IterableDataset requires proper dataloader handling
    dataloader = DataLoader(
        dataset, 
        batch_size=per_device_batch_size, 
        collate_fn=collate,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    # 3. Model initialization
    model = SqueezeformerEncoder(
        input_dim=80, num_layers=16, d_model=256, num_heads=4, 
        ffn_dim=1024, conv_kernel=31, dropout=0.1, vocab_size=vocab_size
    ).to(device)

    if args.init_checkpoint is not None:
        if not os.path.exists(args.init_checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.init_checkpoint}")
        model.load_state_dict(_load_checkpoint_state_dict(args.init_checkpoint, device), strict=True)
        print(f"Initialized model from checkpoint: {args.init_checkpoint}")
    
    # 4. Partial layer freezing
    for name, param in model.named_parameters():
        param.requires_grad = True # train by default
        
        # We want to train LayerNorm parameters, so don't freeze them
        if "norm" in name or "LayerNorm" in name:
            continue
            
        if "conv_subsampling" in name:
            param.requires_grad = False
            
        # Freeze Encoder layers 0-7
        if "layers." in name:
            try:
                # Extract layer index from name, e.g. "layers.0.mha..."
                layer_idx_str = name.split("layers.")[-1].split(".")[0]
                layer_idx = int(layer_idx_str)
                if layer_idx < 8:
                    param.requires_grad = False
            except ValueError:
                pass
                
    # Optimizer only on requires_grad=True
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    
    # Scheduler: Cosine decay with 5% warmup
    estimated_samples = (
        args.estimated_samples
        or dataset_size
        or DEFAULT_ESTIMATED_SAMPLES[args.train_dataset]
    )
    print(f"Using estimated_samples={estimated_samples} for scheduler setup.")

    estimated_steps_per_epoch = estimated_samples // batch_size
    total_steps = estimated_steps_per_epoch * epochs
    warmup_ratio = 0.05
    warmup_steps = int(total_steps * warmup_ratio)
    
    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    scaler = torch.amp.GradScaler(device="cuda", enabled=(device.type == "cuda"))
    ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True).to(device)
    
    # 5. Training Loop
    epoch_offset = _infer_epoch_offset(args.init_checkpoint)
    if epoch_offset > 0:
        print(f"Continuing epoch numbering from checkpoint epoch {epoch_offset}.")

    for epoch in range(1, epochs + 1):
        display_epoch = epoch_offset + epoch
        print(f"--- Epoch {display_epoch} ({epoch}/{epochs} in this run) ---")
        
        # Train
        train_loss = train_epoch(
            model, dataloader, optimizer, scheduler, scaler, ctc_loss, device, 
            feature_extractor, accum_steps=gradient_accumulation, epoch=epoch
        )
        print(f"Epoch {display_epoch} Training Loss: {train_loss:.4f}")
        
        # Evaluate 
        # Evaluate function using Greedy Decoding (to save time during training) or Beam Search
        # beam_search = partial(ctc_beam_search, beam_size=5, blank_id=0)
        # using None defaults to Greedy decoding
        # 
        # Note: For KenLM integration with beam search (future improvement),
        # use torchaudio.models.decoder.ctc_decoder(
        #     lexicon="path/to/lexicon",
        #     tokens="path/to/tokens",
        #     lm="path/to/kenlm.bin",
        #     beam_size=5
        # )
        # TODO: A separate eval dataset should be used for validation here, but using dataloader for now as requested.
        eval_metrics = evaluate(
            model,
            dataloader,
            device,
            tokenizer,
            feature_extractor,
            beam_search_decoder=None,
            return_details=True,
        )
        w_error = eval_metrics["wer"]
        c_error = eval_metrics["cer"]
        
        print(f"Epoch {display_epoch} WER: {w_error:.4f} | CER: {c_error:.4f}")
        
        # Log epoch-level metrics to wandb
        wandb.log({
            "epoch/train_loss": train_loss,
            "epoch/wer": w_error,
            "epoch/cer": c_error,
            "epoch/inference_seconds": eval_metrics["inference_seconds"],
            "epoch/audio_seconds": eval_metrics["audio_seconds"],
            "epoch/rtf": eval_metrics["rtf"],
            "epoch/samples_per_second": eval_metrics["samples_per_second"],
            "epoch/num_samples": eval_metrics["num_samples"],
            "epoch": display_epoch,
            "epoch/lr_epoch_end": scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else scheduler.optimizer.param_groups[0]['lr'],
        })
        
        # Save checkpoint
        checkpoint_path = f"squeezeformer_epoch_{display_epoch}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        
        # Save to wandb as an artifact
        artifact = wandb.Artifact(f"model-epoch-{display_epoch}", type="model")
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)
        
    wandb.finish()

if __name__ == "__main__":
    train()
