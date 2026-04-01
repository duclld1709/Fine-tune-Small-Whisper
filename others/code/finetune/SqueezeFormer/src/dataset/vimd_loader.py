import gzip
import io
import math
import os
import tarfile

import torch
from torch.utils.data import IterableDataset
from datasets import Audio, get_dataset_split_names, load_dataset
from huggingface_hub import hf_hub_download
from .normalization import normalize_vietnamese


def _resolve_split_name(dataset_name, requested_split, config_name=None):
    """
    Resolve common split aliases (val/dev/eval) to actual split names.
    """
    try:
        available_splits = get_dataset_split_names(dataset_name, config_name=config_name)
    except Exception:
        return requested_split

    if requested_split in available_splits:
        return requested_split

    alias_map = {
        "val": ["val", "valid", "validation", "dev"],
        "valid": ["valid", "val", "validation", "dev"],
        "validation": ["validation", "valid", "val", "dev"],
        "dev": ["dev", "valid", "validation", "val"],
        "test": ["test", "eval", "evaluation"],
        "train": ["train", "training"],
    }

    for candidate in alias_map.get(requested_split, []):
        if candidate in available_splits:
            return candidate

    raise ValueError(
        f"Split '{requested_split}' not found for dataset '{dataset_name}'. "
        f"Available splits: {available_splits}"
    )


def _pick_text_column(feature_names, candidate_columns):
    for column_name in candidate_columns:
        if column_name in feature_names:
            return column_name
    raise ValueError(
        "Could not infer transcript/text column. "
        f"Checked {candidate_columns}, available columns: {feature_names}"
    )


def load_audio_waveform(audio_info, target_sr=16000):
    """
    Load audio from a Hugging Face Audio feature or a manually constructed dict.
    Supports:
    - decoded audio objects exposing get_all_samples()
    - dicts with array + sampling_rate
    - dicts with path / bytes when Audio(decode=False) is used
    """
    import soundfile as sf
    from scipy.signal import resample_poly

    def _read_audio(audio_path=None, audio_bytes=None):
        if audio_bytes is not None:
            if hasattr(audio_bytes, "read"):
                audio_bytes = audio_bytes.read()
            audio_source = io.BytesIO(audio_bytes)
        elif audio_path:
            audio_source = audio_path
        else:
            raise ValueError("Expected audio bytes or a filesystem path.")

        waveform_np, source_sr = sf.read(audio_source, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(waveform_np.T.copy())
        return waveform, int(source_sr)

    def _resample_if_needed(waveform, source_sr, desired_sr):
        if source_sr == desired_sr:
            return waveform
        gcd = math.gcd(int(source_sr), int(desired_sr))
        up = desired_sr // gcd
        down = source_sr // gcd
        resampled = resample_poly(waveform.cpu().numpy(), up, down, axis=-1)
        return torch.from_numpy(resampled).to(torch.float32)

    if hasattr(audio_info, "get_all_samples"):
        samples = audio_info.get_all_samples()
        waveform = samples.data.to(torch.float32)
        source_sr = samples.sample_rate
    elif isinstance(audio_info, dict):
        if audio_info.get("array") is not None:
            waveform = torch.as_tensor(audio_info["array"], dtype=torch.float32)
            source_sr = audio_info.get("sampling_rate", target_sr)
        elif audio_info.get("bytes") is not None or audio_info.get("path"):
            waveform, source_sr = _read_audio(
                audio_path=audio_info.get("path"),
                audio_bytes=audio_info.get("bytes"),
            )
        else:
            raise ValueError(f"Unsupported audio dictionary keys: {list(audio_info.keys())}")
    else:
        raise TypeError(f"Unsupported audio_info type: {type(audio_info)!r}")

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = _resample_if_needed(waveform, source_sr, target_sr)

    return waveform.squeeze(0)


def load_hf_asr_dataset(
    dataset_name,
    split,
    config_name=None,
    streaming=True,
    audio_column="audio",
    text_columns=("transcript", "text", "sentence", "transcription", "label"),
    sample_rate=16000,
):
    """
    Load a speech dataset from Hugging Face and standardize to:
    - audio: HF Audio column at sample_rate
    - transcript: normalized target text
    """
    resolved_split = _resolve_split_name(dataset_name, split, config_name=config_name)

    load_kwargs = {
        "path": dataset_name,
        "split": resolved_split,
        "streaming": streaming,
    }
    if config_name is not None:
        load_kwargs["name"] = config_name

    dataset = load_dataset(**load_kwargs)
    feature_names = list(getattr(dataset, "features", {}).keys())

    text_column = _pick_text_column(feature_names, text_columns)

    if audio_column not in feature_names:
        for candidate in ("audio", "speech", "wav"):
            if candidate in feature_names:
                audio_column = candidate
                break
        else:
            raise ValueError(
                f"Could not infer audio column for dataset '{dataset_name}'. "
                f"Available columns: {feature_names}"
            )

    dataset = dataset.cast_column(audio_column, Audio(sampling_rate=sample_rate, decode=False))

    def process_row(example):
        transcript = normalize_vietnamese(example.get(text_column, ""))
        example["transcript"] = transcript
        if audio_column != "audio":
            example["audio"] = example[audio_column]
        return example

    return dataset.map(process_row)


def load_vimd_hf_dataset(
    split="valid",
    dataset_name="nguyendv02/ViMD_Dataset",
    config_name=None,
    streaming=True,
):
    return load_hf_asr_dataset(
        dataset_name=dataset_name,
        split=split,
        config_name=config_name,
        streaming=streaming,
        audio_column="audio",
        text_columns=("transcript", "text", "sentence"),
    )


def load_vivos_hf_dataset(
    split="test",
    dataset_name="AILAB-VNUHCM/vivos",
    config_name=None,
    streaming=True,
):
    try:
        return load_hf_asr_dataset(
            dataset_name=dataset_name,
            split=split,
            config_name=config_name,
            streaming=streaming,
            audio_column="audio",
            text_columns=("sentence", "transcript", "text", "label"),
        )
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" not in str(exc):
            raise
        print(
            "Falling back to manual VIVOS loader because datasets script loading "
            "is not supported in this datasets version."
        )
        return _load_vivos_from_snapshot(dataset_name=dataset_name, split=split)


class _VivosTarIterableDataset(IterableDataset):
    def __init__(self, tar_path, transcript_map, sample_rate=16000):
        super().__init__()
        self.tar_path = tar_path
        self.transcript_map = transcript_map
        self.sample_rate = sample_rate
        self._size = len(transcript_map)

    def __len__(self):
        return self._size

    def __iter__(self):
        yielded = 0
        with tarfile.open(self.tar_path, "r:gz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.lower().endswith(".wav"):
                    continue

                utt_id = os.path.splitext(os.path.basename(member.name))[0]
                transcript = self.transcript_map.get(utt_id) or self.transcript_map.get(utt_id.lower())
                if transcript is None:
                    continue

                audio_file = tar.extractfile(member)
                if audio_file is None:
                    continue

                waveform = load_audio_waveform(
                    {"bytes": audio_file.read()},
                    target_sr=self.sample_rate,
                )

                yielded += 1
                yield {
                    "audio": {
                        "array": waveform.numpy(),
                        "sampling_rate": self.sample_rate,
                    },
                    "transcript": transcript,
                }

                if yielded >= self._size:
                    break


def _parse_vivos_prompts(prompts_path):
    transcript_map = {}
    with gzip.open(prompts_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            utt_id, transcript = parts
            transcript_map[utt_id] = normalize_vietnamese(transcript)
    return transcript_map


def _split_to_vivos_prompt_file(split):
    split = split.lower()
    if split in {"test", "eval", "evaluation", "dev", "valid", "validation", "val"}:
        return "data/prompts-test.txt.gz"
    if split in {"train", "training"}:
        return "data/prompts-train.txt.gz"
    raise ValueError(f"Unsupported VIVOS split '{split}' for manual loading.")


def _load_vivos_from_snapshot(dataset_name, split):
    prompt_file = _split_to_vivos_prompt_file(split)
    prompts_path = hf_hub_download(
        repo_id=dataset_name,
        repo_type="dataset",
        filename=prompt_file,
    )
    tar_path = hf_hub_download(
        repo_id=dataset_name,
        repo_type="dataset",
        filename="data/vivos.tar.gz",
    )
    transcript_map = _parse_vivos_prompts(prompts_path)
    return _VivosTarIterableDataset(tar_path=tar_path, transcript_map=transcript_map, sample_rate=16000)


def load_vivos_transcripts(
    split="train",
    dataset_name="AILAB-VNUHCM/vivos",
):
    """
    Load only normalized VIVOS transcripts from prompts files.
    Useful for tokenizer training without decoding audio.
    """
    prompt_file = _split_to_vivos_prompt_file(split)
    prompts_path = hf_hub_download(
        repo_id=dataset_name,
        repo_type="dataset",
        filename=prompt_file,
    )
    transcript_map = _parse_vivos_prompts(prompts_path)
    return list(transcript_map.values())


def load_vimd_dataset(data_files="Code/src/data/train/*.parquet", streaming=True):
    """
    Use HuggingFace datasets library to load parquet shards efficiently.
    """
    dataset = load_dataset("parquet", data_files=data_files, split="train", streaming=streaming)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000, decode=False))
    
    def process_row(example):
        # Normalize Vietnamese transcripts
        transcript = example.get("transcript", example.get("text", ""))
        example["transcript"] = normalize_vietnamese(transcript)
        return example
    
    dataset = dataset.map(process_row)
    return dataset

def collate_fn(batch, tokenizer):
    waveforms = []
    transcripts = []
    
    for item in batch:
        audio_info = item["audio"]
        waveform_tensor = load_audio_waveform(audio_info, target_sr=16000)
        transcript = item["transcript"]

        waveforms.append(waveform_tensor)
        
        # Tokenize transcript
        tokens = tokenizer.encode(transcript)
        transcripts.append(torch.tensor(tokens, dtype=torch.long))
        
    # Pad waveforms to max time in batch
    waveforms_padded = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
    # 0 is the padding value and CTC blank token
    transcripts_padded = torch.nn.utils.rnn.pad_sequence(transcripts, batch_first=True, padding_value=0)
    
    input_lengths = torch.tensor([w.shape[0] for w in waveforms], dtype=torch.long)
    target_lengths = torch.tensor([t.shape[0] for t in transcripts], dtype=torch.long)
    
    return waveforms_padded, transcripts_padded, input_lengths, target_lengths
