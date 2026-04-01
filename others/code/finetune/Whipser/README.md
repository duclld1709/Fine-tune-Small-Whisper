## Whisper Small Fine-tuning on Kaggle (ViMD Dataset)

Dự án này dùng để **fine-tune mô hình Whisper Small** trên môi trường **Kaggle Notebook**. Toàn bộ cấu trúc và lệnh được tối ưu để có thể clone về và chạy trực tiếp.

---

## Chuẩn bị Dataset

Trước khi chạy notebook, cần thêm dataset vào Kaggle:

1. Dataset ví dụ:
   `ilewanducki/vimd-whisper-autotruncate`

2. Dataset này gồm **3 tập dữ liệu**:

   * `train`
   * `validation`
   * `test`

3. Mỗi tập dữ liệu là kết quả của quá trình:

   * Load audio
   * Trích xuất feature
   * Lưu bằng `save_to_disk()` → sinh ra file **Arrow**

4. Quy trình tạo dataset:

   * Xem script preprocess (xem `src/data_loader.py`)
   * Tạo 3 thư mục dataset
   * Gom vào một thư mục tổng
   * Nén lại và upload lên Kaggle dưới dạng Dataset

---

## Cài đặt Dependencies

Trên Kaggle, PyTorch thường đã được cài sẵn. Tuy nhiên vẫn ghi lại để xử lý xung đột khi cần.

### PyTorch (thường không cần chạy)

```bash
pip install torch==2.8.0+cu126 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### Thư viện còn lại

```bash
pip install transformers datasets evaluate jiwer pyyaml soundfile wandb
pip install --upgrade -q wandb
```

---

## Cấu hình Weights & Biases (wandb)

Do Kaggle có thể dùng bản wandb cũ, cần:

1. **Restart & Clear Output** trong tab Run
2. Đăng nhập bằng API Key:

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

## Chạy Training

### GPU P100 (1 GPU)

```bash
!python train.py
```

### GPU T4 x2 (Multi-GPU)

```bash
!torchrun --nproc_per_node=2 train.py
```

---

## Cấu trúc Thư mục Dự án

```
whisper-finetune/
├── configs/
│   └── config.yaml        # Hyperparameters, training args, đường dẫn dataset
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # Load dataset, audio preprocessing, data collator
│   └── metrics.py         # Tính Word Error Rate (WER)
│
├── train.py               # Entry point để chạy training
├── requirements.txt       # Danh sách thư viện
└── README.md              # Tài liệu hướng dẫn
```

---

## Ghi chú

* Dataset cần **feature extraction trước**, không dùng audio raw.
* wandb bắt buộc nếu muốn theo dõi loss / WER trực quan.
* Multi-GPU chỉ cần khi dataset lớn hoặc muốn rút ngắn thời gian train.
* `config.yaml` là nơi điều chỉnh:

  * batch size
  * learning rate
  * số epoch
  * đường dẫn dataset
  * logging / save steps

