from ..base_model import BaseASRModel
import torch
from torch import nn
import sentencepiece as spm
from .model_scratch import SqueezeformerEncoder
import soundfile as sf
import math
from scipy.signal import resample_poly
import torchaudio

import os

class SqueezeFormer(BaseASRModel):
    def __init__(self, checkpoint_path=None):
        if checkpoint_path == None:
            checkpoint_path = r"D:\Duc_Data\Study\FPT_University_Course\SPRING26_Semester_7\DAT301\Group8_DAT301\code\demo\checkpoints\squeezeformer"
        super().__init__(checkpoint_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        vocab_size = 2000
        self.tokenizer = Tokenizer(os.path.join(self.checkpoint_path, "vimd_unigram.model"))
        self.model = SqueezeformerEncoder(
            input_dim=80, num_layers=16, d_model=256, num_heads=4, 
            ffn_dim=1024, conv_kernel=31, dropout=0.0, vocab_size=vocab_size).to(self.device)
        state_dict = torch.load(os.path.join(self.checkpoint_path, "squeezeformer_epoch_31.pt"), map_location=self.device, weights_only=True)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                new_state_dict[k[10:]] = v
            else:
                new_state_dict[k] = v
                
        self.model.load_state_dict(new_state_dict)
        self.model.eval()

    def transcribe(self, audio_path: str) -> str:
        target_sr = 16000
        feature_extractor = LogMelExtractor(
                                    sample_rate=target_sr, n_fft=400, hop_length=160, win_length=400, n_mels=80, f_min=0, f_max=8000
                                ).to(self.device)
        # Load audio
        waveform_np, source_sr = sf.read(audio_path, dtype="float32", always_2d=True)
        
        # (channels, time)
        waveform = torch.from_numpy(waveform_np.T)

        # Convert stereo -> mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample nếu cần
        if source_sr != target_sr:
            gcd = math.gcd(source_sr, target_sr)
            up = target_sr // gcd
            down = source_sr // gcd
            waveform = torch.from_numpy(
                resample_poly(waveform.numpy(), up, down, axis=-1)
            ).float()

        # Extract audio
        waveform_tensor = waveform.squeeze(0)

        # Add batch dim
        waveform_tensor = waveform_tensor.unsqueeze(0).to(self.device)
        input_lengths = torch.tensor([waveform_tensor.shape[1]], dtype=torch.long).to(self.device)
        
        # Forward Pass
        with torch.no_grad():
            features, feat_lengths = feature_extractor(waveform_tensor, input_lengths)
            
            with torch.amp.autocast(device_type="cuda"):
                logits, out_lengths = self.model(features, feat_lengths)
                
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            
            # Greedy Decoding
            preds = log_probs.argmax(dim=-1)[0] # Take first item in batch
            
            # CTC Decode
            p = preds.tolist()
            collapsed = []
            prev = -1
            for t in p:
                if t != 0 and t != prev: # 0 is blank
                    collapsed.append(t)
                prev = t
                
            predicted_text = self.tokenizer.decode(collapsed)
            # Post-process Spm special characters
            predicted_text = predicted_text.replace(' ', ' ').strip()
        return predicted_text

def unload(self):
    import gc

    # 1. Model
    if hasattr(self, "model") and self.model is not None:
        try:
            self.model.to("cpu")
        except:
            pass
        del self.model
        self.model = None

    # 2. Tokenizer (sentencepiece cũng giữ memory)
    if hasattr(self, "tokenizer") and self.tokenizer is not None:
        del self.tokenizer
        self.tokenizer = None

    # 3. Dọn các object khác nếu bạn có giữ lại (optional)
    if hasattr(self, "feature_extractor"):
        del self.feature_extractor
        self.feature_extractor = None

    # 4. Python GC
    gc.collect()

    # 5. GPU cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

class Tokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        
    def encode(self, text):
        return self.sp.encode_as_ids(text)
        
    def decode(self, ids):
        # Convert list of integers or tensor to python list
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return self.sp.decode_ids(ids)
        
    @property
    def vocab_size(self):
        return self.sp.vocab_size()
    
class LogMelExtractor(nn.Module):
    def __init__(self, 
                 sample_rate=16000, 
                 n_fft=400, 
                 hop_length=160, 
                 win_length=400, 
                 n_mels=80, 
                 f_min=0.0, 
                 f_max=8000.0):
        super().__init__()
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            normalized=True
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, waveform, lengths=None):
        """
        waveform: (batch, time) or (1, time)
        returns: (batch, time, n_mels)
        """
        mel = self.mel_spectrogram(waveform)
        log_mel = self.amplitude_to_db(mel)
        # LogMel Spectrogram output is typically (batch, n_mels, time)
        # Transpose to (batch, time, n_mels)
        if log_mel.dim() == 3:
            log_mel = log_mel.transpose(1, 2)
        elif log_mel.dim() == 2:
            log_mel = log_mel.transpose(0, 1)
            
        if lengths is not None:
            # hop_length is usually how much time compresses
            lengths = (lengths // self.mel_spectrogram.hop_length) + 1
            return log_mel, lengths
            
        return log_mel

class SpecAugment(nn.Module):
    def __init__(self, time_mask=2, time_mask_width=40, freq_mask=2, freq_mask_width=15):
        super().__init__()
        self.time_mask = time_mask
        self.freq_mask = freq_mask
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_width)
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_width)

    def forward(self, spec):
        """
        spec: (batch, time, n_mels)
        """
        # torchaudio masks expect (batch, n_mels, time)
        spec = spec.transpose(-1, -2)
        
        for _ in range(self.time_mask):
            spec = self.time_masking(spec)
        for _ in range(self.freq_mask):
            spec = self.freq_masking(spec)
            
        spec = spec.transpose(-1, -2)
        return spec
