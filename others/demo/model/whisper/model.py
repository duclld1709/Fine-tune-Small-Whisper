from ..base_model import BaseASRModel
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
import torchaudio
import os

class Whisper(BaseASRModel):
    def __init__(self, checkpoint_path=None, version="small"):
        if checkpoint_path == None:
            checkpoint_path = r"D:\Duc_Data\Study\FPT_University_Course\SPRING26_Semester_7\DAT301\Group8_DAT301\code\demo\checkpoints\whisper"
        super().__init__(checkpoint_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.version = version

    def load_model(self):
        self.processor = WhisperProcessor.from_pretrained(os.path.join(self.checkpoint_path, self.version), language="vi", task="transcribe")
        self.model = WhisperForConditionalGeneration.from_pretrained(os.path.join(self.checkpoint_path, self.version)).to(self.device)
        self.model.eval()

    def transcribe(self, audio_path: str) -> str:
        TARGET_SR = 16000
        array, sr = sf.read(audio_path, dtype="float32")
        array = torch.from_numpy(array)
        if array.ndim > 1:
            array = array.mean(dim=1)
        if sr != TARGET_SR:
            array = torchaudio.functional.resample(array, sr, TARGET_SR)
        inputs = self.processor(
            array,
            sampling_rate=TARGET_SR,
            return_tensors="pt"
        )

        input_features = inputs.input_features.to(self.device)
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]

        return transcription.lower()

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

    # 2. Processor (WhisperProcessor khá nặng)
    if hasattr(self, "processor") and self.processor is not None:
        del self.processor
        self.processor = None

    # 3. Dọn các tensor tạm nếu có giữ reference
    if hasattr(self, "input_features"):
        del self.input_features
        self.input_features = None

    # 4. Garbage Collector
    gc.collect()

    # 5. GPU cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()