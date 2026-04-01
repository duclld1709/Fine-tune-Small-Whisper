from abc import ABC, abstractmethod

class BaseASRModel(ABC):
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.model = None

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        pass

    def unload(self):
        pass