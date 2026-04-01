from model.squeezeformer.model import SqueezeFormer
from model.wav2vec.model import Wav2Vec
from model.whisper.model import Whisper
from model.qwenasr.model import QwenASR
import time

class ModelManager:
    def __init__(self):
        self.current_model_name = None
        self.current_model = None

    def load(self, model_name: str):
        if self.current_model_name == model_name:
            return self.current_model

        # unload model cũ
        if self.current_model is not None:
            print(f"[INFO] Unloading {self.current_model_name}...")
            self.current_model.unload()

        print(f"[INFO] Loading {model_name}...")

        # parse model
        if model_name.startswith("Whisper"):
            version = model_name.split("-")[1]  # tiny/base/small
            self.current_model = Whisper(version=version)

        elif model_name == "Wav2Vec 2.0 Base":
            self.current_model = Wav2Vec()

        elif model_name == "SqueezeFormer Medium":
            self.current_model = SqueezeFormer()

        elif model_name == "QwenASR 0.6B":
            self.current_model = QwenASR()

        else:
            raise ValueError("Unknown model")

        self.current_model.load_model()
        self.current_model_name = model_name

        return self.current_model

import gradio as gr

manager = ModelManager()

def transcribe_fn(audio, model_name):
    if audio is None:
        return "No audio provided!"

    # Load model
    model = manager.load(model_name)

    import soundfile as sf
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio[1], audio[0])

        start_time = time.perf_counter()
        result = model.transcribe(tmp.name)
        end_time = time.perf_counter()

        inference_time = end_time - start_time

    return f"{result}\n\n⏱️ Inference time: {inference_time:.3f} seconds"


with gr.Blocks(theme=gr.themes.Soft(), title="ASR Demo") as demo:

    gr.Markdown(
        """
        # 🎤 ASR Multi-Model Demo
        Select model → Upload or Record → Generate transcription
        """
    )

    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=[
                "Wav2Vec 2.0 Base",
                "Whisper-tiny",
                "Whisper-base",
                "Whisper-small",
                "SqueezeFormer Medium",
                "QwenASR 0.6B"
            ],
            value="QwenASR 0.6B",
            label="🤖 Select Model"
        )

    with gr.Row():
        audio_input = gr.Audio(
            sources=["upload", "microphone"],
            type="numpy",
            label="🎧 Input Audio"
        )

    generate_btn = gr.Button("🚀 Transcribe", variant="primary")

    output_text = gr.Textbox(
        label="📝 Output Text",
        lines=5
    )

    status_text = gr.Textbox(
        label="⚙️ Status",
        value="Ready",
        interactive=False
    )

    # Khi click button
    generate_btn.click(
        fn=transcribe_fn,
        inputs=[audio_input, model_dropdown],
        outputs=output_text
    )

    # Khi đổi model → unload tự động
    def on_model_change(model_name):
        if manager.current_model is not None:
            manager.current_model.unload()
            manager.current_model = None
            manager.current_model_name = None
        return f"Switched to {model_name} (old model unloaded)"

    model_dropdown.change(
        fn=on_model_change,
        inputs=model_dropdown,
        outputs=status_text
    )


demo.launch()