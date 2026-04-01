## Whisper Small Fine-tuning on Kaggle (ViMD Dataset)

This project is designed to **fine-tune the Whisper Small model** in a **Kaggle Notebook environment**. The structure and commands are optimized so you can clone the repository and run it directly.

---

## Dataset Preparation

Before running the notebook, you need to add the dataset to Kaggle:

1. Example dataset:
   `ilewanducki/vimd-whisper-autotruncate`

2. The dataset contains **three splits**:

   * `train`
   * `validation`
   * `test`

3. Each split is generated through:

   * Loading audio
   * Extracting features
   * Saving with `save_to_disk()` → produces **Arrow files**

4. Dataset creation workflow:

   * Refer to the preprocessing script (`src/data_loader.py`)
   * Create three dataset directories
   * Combine them into a single root folder
   * Compress and upload to Kaggle as a Dataset

---

## Install Dependencies

On Kaggle, PyTorch is usually preinstalled. The command below is provided in case of version conflicts.

### PyTorch (usually not required)

```bash
pip install torch==2.8.0+cu126 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### Other Libraries

```bash
pip install transformers datasets evaluate jiwer pyyaml soundfile wandb
pip install --upgrade -q wandb
```

---

## Configure Weights & Biases (wandb)

Since Kaggle may use an older version of wandb:

1. Click **Restart & Clear Output** in the Run tab
2. Log in using your API key:

```python
import os
import wandb
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
wandb_api = user_secrets.get_secret("WANDB_API_KEY")

wandb.login(key=wandb_api)

os.environ["WANDB_PROJECT"] = "small-full-autotruncate"
os.environ["WANDB_ENTITY"] = "dat301_ai1802"
```

---

## Clone Repository

```bash
!git clone https://github.com/duclld1709/Fine-tune-Small-Whisper.git
%cd Fine-tune-Small-Whisper
!ls
```

---

## Run Training

### Single GPU (P100)

```bash
!python train.py
```

### Multi-GPU (2× T4)

```bash
!torchrun --nproc_per_node=2 train.py
```

---

## Project Structure

```
whisper-finetune/
├── configs/
│   └── config.yaml        # Hyperparameters, training args, dataset paths
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # Dataset loading, audio preprocessing, data collator
│   └── metrics.py         # Word Error Rate (WER) computation
│
├── train.py               # Training entry point
├── requirements.txt       # Dependencies list
├── report.pdf             # Research-Based report of the whole projects
├── others/                # Demo, fine-tune and eval of others models from team members
└── README.md              # Documentation
```

---

## Notes

* The dataset must be **preprocessed with feature extraction** (raw audio is not used).
* wandb is required if you want to track loss and WER visually.
* Multi-GPU is recommended for large datasets or faster training.
* `config.yaml` is where you configure:

  * batch size
  * learning rate
  * number of epochs
  * dataset paths
  * logging and checkpoint saving steps
