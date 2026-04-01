
import torch
import numpy as np
import gradio as gr
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# ── 1. Load Model ─────────────────────────────────────────
MODEL_PATH = r"D:\wav2vec2-vimd-vn"

print(f"Loading model từ: {MODEL_PATH}")
processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
model     = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = model.to(device)
print(f"✅ Model loaded on: {device}")


# ── 2. Inference function ─────────────────────────────────
def transcribe(audio):
    if audio is None:
        return "⚠️ Vui lòng upload file hoặc ghi âm trước!"

    if isinstance(audio, str):
        speech, sr = librosa.load(audio, sr=16_000)
    else:
        sr, speech = audio
        speech = speech.astype(np.float32)

        if len(speech.shape) > 1:
            speech = speech.mean(axis=1)

        if sr != 16_000:
            speech = librosa.resample(speech, orig_sr=sr, target_sr=16_000)

    if speech.max() > 1.0:
        speech = speech / 32768.0

    duration = len(speech) / 16_000
    if duration < 0.5:
        return "⚠️ Audio quá ngắn, vui lòng nói ít nhất 0.5 giây!"
    if duration > 60:
        return "⚠️ Audio quá dài, vui lòng giới hạn dưới 60 giây!"

    inputs = processor(
        speech,
        sampling_rate=16_000,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits

    pred_ids      = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(pred_ids)[0]

    return transcription.strip() if transcription.strip() else "⚠️ Không nhận dạng được âm thanh"


# ── 3. Gradio Interface (Gradio 3.x) ──────────────────────
with gr.Blocks(
    title="ASR Tiếng Việt",
    theme=gr.themes.Soft(),     
) as demo:

    gr.Markdown("""
    # 🎙️ Nhận dạng giọng nói Tiếng Việt
    
    """)

    with gr.Tabs():

        with gr.Tab("📁 Upload File"):
            gr.Markdown("Hỗ trợ: **MP3, WAV, FLAC, OGG, M4A**")
            with gr.Row():
                with gr.Column(scale=1):
                    audio_upload = gr.Audio(
                        label="Upload file âm thanh",
                        type="filepath",
                        # ✅ bỏ sources= không có trong Gradio 3.x
                    )
                    btn_upload = gr.Button("🔍 Nhận dạng", variant="primary", size="lg")
                    btn_clear_upload = gr.Button("🗑️ Xóa", variant="secondary", size="sm")
                with gr.Column(scale=1):
                    output_upload = gr.Textbox(
                        label="Kết quả nhận dạng",
                        placeholder="Kết quả sẽ hiển thị ở đây...",
                        lines=8,
                        show_copy_button=True,
                    )

            btn_upload.click(fn=transcribe, inputs=audio_upload, outputs=output_upload)
            btn_clear_upload.click(fn=lambda: (None, ""), outputs=[audio_upload, output_upload])

        with gr.Tab("🎤 Ghi âm trực tiếp"):
            gr.Markdown("Nhấn **Record** để bắt đầu, nhấn **⏹ Stop** để dừng")
            with gr.Row():
                with gr.Column(scale=1):
                    audio_record = gr.Audio(
                        label="Ghi âm",
                        type="numpy",
                        source="microphone",    
                    )
                    btn_record = gr.Button("🔍 Nhận dạng", variant="primary", size="lg")
                    btn_clear_record = gr.Button("🗑️ Xóa", variant="secondary", size="sm")
                with gr.Column(scale=1):
                    output_record = gr.Textbox(
                        label="Kết quả nhận dạng",
                        placeholder="Kết quả sẽ hiển thị ở đây...",
                        lines=8,
                        show_copy_button=True,
                    )

            btn_record.click(fn=transcribe, inputs=audio_record, outputs=output_record)
            btn_clear_record.click(fn=lambda: (None, ""), outputs=[audio_record, output_record])

    gr.Markdown("""
    ---
    ### 💡 Lưu ý để có kết quả tốt nhất
    - 🎙️ Nói rõ ràng, tốc độ vừa phải
    - 🔇 Môi trường ít tiếng ồn
    - ⏱️ Độ dài tối ưu: **5 - 30 giây**
    """)


# ── 4. Chạy app ───────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )