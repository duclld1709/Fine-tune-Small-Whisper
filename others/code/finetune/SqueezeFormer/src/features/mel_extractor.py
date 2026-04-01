import torch
import torch.nn as nn
import torchaudio

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
