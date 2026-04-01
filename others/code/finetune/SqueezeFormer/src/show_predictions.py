import argparse
import glob
import random

import torch

from src.dataset.vimd_loader import load_audio_waveform, load_vimd_hf_dataset, load_vivos_hf_dataset
from src.features.mel_extractor import LogMelExtractor
from src.models.squeezeformer import SqueezeformerEncoder
from src.tokenizer.train_tokenizer import Tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Show random ground-truth vs prediction samples")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["vimd_test", "vivos_test"],
        choices=["vimd_test", "vivos_test"],
        help="Which datasets to sample from",
    )
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples per dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path (default: latest)")
    parser.add_argument("--tokenizer-model", type=str, default="vimd_unigram.model")
    parser.add_argument("--vocab-size", type=int, default=2000)
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu, or auto")
    parser.add_argument("--no-streaming", action="store_true", help="Disable HF streaming mode")
    parser.add_argument("--vimd-dataset", type=str, default="nguyendv02/ViMD_Dataset")
    parser.add_argument("--vimd-config", type=str, default=None)
    parser.add_argument("--vivos-dataset", type=str, default="AILAB-VNUHCM/vivos")
    parser.add_argument("--vivos-config", type=str, default=None)
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


def extract_waveform(audio_info, target_sr=16000):
    return load_audio_waveform(audio_info, target_sr=target_sr)


def greedy_decode_ids(log_probs):
    preds = log_probs.argmax(dim=-1)[0].tolist()
    collapsed = []
    prev = -1
    for token_id in preds:
        if token_id != 0 and token_id != prev:
            collapsed.append(token_id)
        prev = token_id
    return collapsed


def predict_text(sample, model, feature_extractor, tokenizer, device):
    waveform = extract_waveform(sample["audio"]).unsqueeze(0).to(device)
    input_lengths = torch.tensor([waveform.shape[1]], dtype=torch.long, device=device)

    with torch.no_grad():
        features, feat_lengths = feature_extractor(waveform, input_lengths)
        autocast_device_type = "cuda" if device.type == "cuda" else "cpu"
        with torch.amp.autocast(device_type=autocast_device_type, enabled=(device.type == "cuda")):
            logits, _ = model(features, feat_lengths)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    token_ids = greedy_decode_ids(log_probs)
    return tokenizer.decode(token_ids).strip()


def reservoir_sample(dataset, k, seed):
    rng = random.Random(seed)
    selected = []
    for idx, item in enumerate(dataset):
        if idx < k:
            selected.append(item)
            continue
        j = rng.randint(0, idx)
        if j < k:
            selected[j] = item
    return selected


def sample_random_examples(dataset, num_samples, seed):
    if hasattr(dataset, "shuffle") and hasattr(dataset, "take"):
        try:
            # Keep buffer moderate so streaming datasets return quickly.
            shuffled = dataset.shuffle(seed=seed, buffer_size=max(100, num_samples * 10))
            return list(shuffled.take(num_samples))
        except Exception:
            pass

    if hasattr(dataset, "select") and hasattr(dataset, "__len__"):
        try:
            total = len(dataset)
            if total == 0:
                return []
            rng = random.Random(seed)
            indices = list(range(total))
            rng.shuffle(indices)
            indices = indices[: min(num_samples, total)]
            return [dataset[i] for i in indices]
        except Exception:
            pass

    return reservoir_sample(dataset, num_samples, seed)


def build_dataset(target, args, streaming):
    if target == "vimd_test":
        return load_vimd_hf_dataset(
            split="test",
            dataset_name=args.vimd_dataset,
            config_name=args.vimd_config,
            streaming=streaming,
        )
    if target == "vivos_test":
        return load_vivos_hf_dataset(
            split="test",
            dataset_name=args.vivos_dataset,
            config_name=args.vivos_config,
            streaming=streaming,
        )
    raise ValueError(f"Unsupported target: {target}")


def main():
    args = parse_args()
    streaming = not args.no_streaming
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = args.checkpoint if args.checkpoint else find_latest_checkpoint()
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

    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint_path}")

    for target_idx, target in enumerate(args.targets):
        print(f"\n=== {target} ({args.num_samples} random samples) ===")
        dataset = build_dataset(target, args, streaming=streaming)
        print("Sampling examples...")
        examples = sample_random_examples(dataset, args.num_samples, seed=args.seed + target_idx)

        if not examples:
            print("No samples found.")
            continue

        for i, sample in enumerate(examples, start=1):
            gt = str(sample.get("transcript", "")).strip()
            pred = predict_text(sample, model, feature_extractor, tokenizer, device)
            print(f"\nSample {i}")
            print(f"ground truth: {gt}")
            print(f"predict : {pred}")


if __name__ == "__main__":
    main()
