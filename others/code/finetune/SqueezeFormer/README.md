# Squeezeformer ASR for Vietnamese

This repository contains a PyTorch implementation of a Squeezeformer-based automatic speech recognition (ASR) pipeline for Vietnamese speech. The codebase supports:

- training on ViMD from local parquet shards
- training or fine-tuning on VIVOS from Hugging Face
- evaluation on ViMD validation/test and VIVOS test
- sample-level inference with ground truth, prediction, and per-sample inference time

The project currently uses:

- a 16-layer Squeezeformer encoder
- SentencePiece tokenization
- log-Mel frontend features
- CTC training and greedy decoding

## Project Layout

```text
.
|-- run_inference_samples.py     # sample inference runner
|-- requirements.txt            # main project dependencies
|-- squeezeformer_epoch_*.pt    # saved checkpoints
|-- vimd_unigram.model          # SentencePiece model
|-- src/
|   |-- train.py                # training entrypoint
|   |-- evaluate.py             # evaluation entrypoint
|   |-- show_predictions.py     # random GT vs prediction helper
|   |-- test_sample.py          # quick single-script sanity check
|   |-- dataset/
|   |-- features/
|   |-- models/
|   |-- tokenizer/
|   `-- training/
`-- wandb/                      # experiment logs
```

## Requirements

- Python 3.10+
- Conda recommended on Windows
- CUDA GPU recommended for training and faster inference

The main dependencies are listed in `requirements.txt`.

## Setup

### 1. Create an environment

PowerShell:

```powershell
conda create -n conda_env python=3.12 -y
conda init powershell
```

Close and reopen PowerShell, then activate:

```powershell
conda activate conda_env
```

If `conda activate` still does not work, you can run everything in this README with `conda run -n conda_env ...` instead.

### 2. Install dependencies

From the repository root:

```powershell
pip install -r requirements.txt
```


## Datasets

### ViMD

Training on ViMD expects local parquet shards. The default path used by `src.train` is:

```text
Code/src/data/train/*.parquet
```

You can override it with `--vimd-data-files`.

For evaluation and inference, the code uses the Hugging Face dataset:

- `nguyendv02/ViMD_Dataset`

### VIVOS

VIVOS is loaded from Hugging Face:

- `AILAB-VNUHCM/vivos`

If your installed `datasets` version does not support the legacy VIVOS dataset script directly, the code automatically falls back to the snapshot files and still works.

## Training

Run all commands from the repository root.

### Train on ViMD

```powershell
conda run -n conda_env python -m src.train `
  --train-dataset vimd `
  --vimd-data-files "Code/src/data/train/*.parquet" `
  --epochs 20 `
  --batch-size 16 `
  --gradient-accumulation 4 `
  --learning-rate 3e-4 `
  --vocab-size 2000
```

This will:

- train a SentencePiece tokenizer if one does not already exist
- train the Squeezeformer model
- save checkpoints like `squeezeformer_epoch_1.pt`

### Train on VIVOS from scratch

```powershell
conda run -n conda_env python -m src.train `
  --train-dataset vivos `
  --vivos-split train `
  --epochs 20 `
  --batch-size 16 `
  --gradient-accumulation 4 `
  --learning-rate 3e-4 `
  --vocab-size 2000
```

### Fine-tune on VIVOS from an existing ViMD checkpoint

This is the pattern already used in the workspace:

```powershell
conda run -n conda_env python -m src.train `
  --train-dataset vivos `
  --vivos-split train `
  --init-checkpoint squeezeformer_epoch_20.pt `
  --tokenizer-model vimd_unigram.model `
  --epochs 20 `
  --estimated-samples 11660
```

## Evaluation

Evaluate on any combination of:

- `vimd_val`
- `vimd_test`
- `vivos_test`

Example:

```powershell
conda run -n conda_env python -m src.evaluate `
  --targets vimd_val vimd_test vivos_test `
  --checkpoint squeezeformer_epoch_31.pt `
  --batch-size 8 `
  --no-wandb
```

If you omit `--checkpoint`, the latest `squeezeformer_epoch_*.pt` file in the root folder is used automatically.

## Sample Inference

To print 5 random samples from ViMD test and VIVOS test, including:

- ground truth
- model prediction
- per-sample inference time

run:

```powershell
conda run -n conda_env python run_inference_samples.py
```

Useful options:

```powershell
conda run -n conda_env python run_inference_samples.py `
  --checkpoint squeezeformer_epoch_31.pt `
  --num-samples 5 `
  --targets vimd_test vivos_test
```

Model checkpoints:

https://drive.google.com/drive/folders/1TD0s-1LJNpd3UqX2o3PZUCkw5iIbS3zW?usp=sharing
