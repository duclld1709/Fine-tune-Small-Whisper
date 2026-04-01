from ..base_model import BaseASRModel
import torch
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf
import librosa
import gc

class Wav2Vec(BaseASRModel):
    def __init__(self, checkpoint_path=None):
        if checkpoint_path == None:
            checkpoint_path = r"D:\Duc_Data\Study\FPT_University_Course\SPRING26_Semester_7\DAT301\Group8_DAT301\code\demo\checkpoints\wav2vec"
        super().__init__(checkpoint_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        self.model = Wav2Vec2ForCTC.from_pretrained(self.checkpoint_path)
        self.processor = Wav2Vec2Processor.from_pretrained(self.checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

    def transcribe(self, audio_path: str) -> str:
        array, sr = sf.read(audio_path)
        array = array.astype(np.float32)

        if len(array.shape) > 1:
            array = array.mean(axis=1)

        if sr != 16000:
            array = librosa.resample(array, orig_sr=sr, target_sr=16000)

        inputs = self.processor(
            array,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        input_values = inputs.input_values
        input_values = input_values.to(self.device)

        with torch.no_grad():
            logits = self.model(input_values).logits

        pred_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.decode(pred_ids[0])

        return transcription
    def unload(self):
        if hasattr(self, "model") and self.model is not None:
            self.model.to("cpu")
            del self.model
            self.model = None

        if hasattr(self, "processor") and self.processor is not None:
            del self.processor
            self.processor = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()