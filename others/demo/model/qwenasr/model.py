from ..base_model import BaseASRModel
import torch
import gc
from qwen_asr import Qwen3ASRModel

class QwenASR(BaseASRModel):
    def __init__(self, checkpoint_path=None):
        if checkpoint_path == None:
            checkpoint_path = r"D:\Duc_Data\Study\FPT_University_Course\SPRING26_Semester_7\DAT301\Group8_DAT301\code\demo\checkpoints\qwenasr\snapshots\5eb144179a02acc5e5ba31e748d22b0cf3e303b0"
        super().__init__(checkpoint_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
        SAMPLING_RATE = 16000
        self.model = Qwen3ASRModel.from_pretrained(
            self.checkpoint_path,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            max_inference_batch_size=32,
            max_new_tokens=256,
        )

    def transcribe(self, audio_path: str) -> str:
        results = self.model.transcribe(audio=audio_path, language=None)
        output_text = results[0].text
        return output_text

    def unload(self):

        if hasattr(self, "model") and self.model is not None:
            try:
                if hasattr(self.model, "to"):
                    self.model.to("cpu")
            except:
                pass

            del self.model
            self.model = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
