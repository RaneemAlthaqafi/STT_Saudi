"""
Saudi STT Demo — MasriSwitch-Gemma3n Transcriber
ZATCA Brand Identity Theme
Run: python app_gradio.py
"""

import os
import sys
import time
import torch
import numpy as np
import gradio as gr

# ──────────────────────────────────────────────────────────────
# Model loading (singleton)
# ──────────────────────────────────────────────────────────────
MODEL = None
PROCESSOR = None


def load_model():
    global MODEL, PROCESSOR
    if MODEL is not None:
        return MODEL, PROCESSOR

    from transformers import AutoProcessor, Gemma3nForConditionalGeneration

    print("Loading MasriSwitch-Gemma3n-Transcriber-v1...")
    PROCESSOR = AutoProcessor.from_pretrained("google/gemma-3n-E4B-it")
    MODEL = Gemma3nForConditionalGeneration.from_pretrained(
        "oddadmix/MasriSwitch-Gemma3n-Transcriber-v1",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    ).eval()

    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"  GPU memory: {mem:.1f} GB")
    print("  Model loaded!")
    return MODEL, PROCESSOR


def transcribe(audio_path):
    """Transcribe audio file with MasriSwitch-Gemma3n."""
    if audio_path is None:
        return "الرجاء رفع ملف صوتي أو فيديو", "", ""

    model, processor = load_model()

    # Load audio
    import librosa
    audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = len(audio_array) / sr

    # Build chat messages
    messages = [
        {"role": "system", "content": [
            {"type": "text", "text": "You are an assistant that transcribes speech accurately."}
        ]},
        {"role": "user", "content": [
            {"type": "audio", "audio": audio_array},
            {"type": "text", "text": "Please transcribe this audio."}
        ]},
    ]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
    )

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    moved = {}
    for k, v in inputs.items():
        if hasattr(v, 'to'):
            v = v.to(device)
            if v.is_floating_point():
                v = v.to(dtype)
        moved[k] = v
    inputs = moved
    input_len = inputs["input_ids"].shape[-1]

    # Generate
    t0 = time.time()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    elapsed = time.time() - t0

    transcript = processor.decode(output[0][input_len:], skip_special_tokens=True).strip()

    # Stats
    words = len(transcript.split()) if transcript else 0
    chars = len(transcript)
    rtf = elapsed / duration if duration > 0 else 0
    speed = f"{1/rtf:.1f}x" if rtf > 0 else "N/A"

    stats_md = f"""
| المقياس | القيمة |
|---------|--------|
| مدة المقطع | {format_duration(duration)} |
| عدد الكلمات | {words} |
| عدد الأحرف | {chars} |
| وقت المعالجة | {elapsed:.1f} ث |
| سرعة المعالجة | {speed} من الوقت الحقيقي |
"""

    info = f"مدة: {format_duration(duration)} | كلمات: {words} | سرعة: {speed}"
    return transcript, stats_md, info


def format_duration(sec):
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m}:{s:02d}" if m > 0 else f"{s} ثانية"


# ──────────────────────────────────────────────────────────────
# ZATCA Theme CSS
# ──────────────────────────────────────────────────────────────
ZATCA_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700;800;900&display=swap');

/* Global */
.gradio-container {
    font-family: 'Tajawal', sans-serif !important;
    max-width: 900px !important;
    margin: 0 auto !important;
}

/* Header */
.hero-banner {
    background: linear-gradient(160deg, #0f1a2e 0%, #1D3761 30%, #2053A4 60%, #0996D4 100%);
    border-radius: 16px;
    padding: 40px 32px;
    text-align: center;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}

.hero-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        linear-gradient(3deg, transparent 48%, rgba(79,187,189,0.06) 49%, rgba(79,187,189,0.06) 51%, transparent 52%),
        linear-gradient(-2deg, transparent 48%, rgba(98,179,79,0.05) 49%, rgba(98,179,79,0.05) 51%, transparent 52%),
        linear-gradient(1.5deg, transparent 48%, rgba(9,150,212,0.04) 49%, rgba(9,150,212,0.04) 51%, transparent 52%);
    pointer-events: none;
}

.hero-banner h1 {
    color: #fff;
    font-size: 36px;
    font-weight: 800;
    margin: 0 0 8px;
    position: relative;
}

.hero-banner .accent {
    background: linear-gradient(135deg, #4FBBBD, #62B34F);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-banner p {
    color: rgba(255,255,255,0.6);
    font-size: 16px;
    margin: 0;
    position: relative;
}

.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    padding: 6px 18px;
    border-radius: 50px;
    color: #4FBBBD;
    font-size: 13px;
    font-weight: 500;
    margin-bottom: 16px;
    position: relative;
}

/* Cards */
.card-section {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 20px;
    box-shadow: 0 4px 24px rgba(29,55,97,0.06);
    position: relative;
    overflow: hidden;
}

.card-section::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(135deg, #4FBBBD, #62B34F);
}

.card-section h3 {
    color: #1D3761;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 16px;
}

/* Button override */
.primary-btn, button.primary {
    background: linear-gradient(135deg, #1D3761, #2053A4, #0996D4) !important;
    border: none !important;
    color: #fff !important;
    font-family: 'Tajawal', sans-serif !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    padding: 14px 32px !important;
    border-radius: 12px !important;
    cursor: pointer !important;
    transition: all 0.3s !important;
}

.primary-btn:hover, button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(32,83,164,0.3) !important;
}

/* Transcript output */
.transcript-box textarea {
    font-family: 'Tajawal', sans-serif !important;
    font-size: 20px !important;
    line-height: 2 !important;
    direction: rtl !important;
    color: #1D3761 !important;
    border: 2px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 20px !important;
}

.transcript-box textarea:focus {
    border-color: #0996D4 !important;
    box-shadow: 0 0 0 3px rgba(9,150,212,0.1) !important;
}

/* Stats table */
.stats-table {
    border-radius: 12px;
    overflow: hidden;
}

.stats-table table {
    font-family: 'Tajawal', sans-serif !important;
}

.stats-table th {
    background: #1D3761 !important;
    color: #fff !important;
    font-weight: 700 !important;
}

.stats-table td {
    color: #1D3761 !important;
    font-weight: 500 !important;
}

/* Footer */
.footer-text {
    text-align: center;
    color: #575756;
    font-size: 13px;
    padding: 16px;
    border-top: 1px solid #e2e8f0;
    margin-top: 12px;
}

/* Audio component */
.audio-input label, .audio-output label {
    font-family: 'Tajawal', sans-serif !important;
    font-weight: 700 !important;
    color: #1D3761 !important;
}

/* Dark mode support */
.dark .card-section { background: #1a1a2e; border-color: #2a2a4a; }
.dark .hero-banner { border-color: #2a2a4a; }
"""

# ──────────────────────────────────────────────────────────────
# Build Gradio UI
# ──────────────────────────────────────────────────────────────

HERO_HTML = """
<div class="hero-banner">
    <div class="hero-badge">نظام تحويل الصوت إلى نص</div>
    <h1>تحويل <span class="accent">الصوت</span> إلى نص عربي</h1>
    <p>ارفع مقطع صوتي أو فيديو وسيتم تحويله تلقائياً إلى نص باللهجة السعودية</p>
</div>
"""

FOOTER_HTML = """
<div class="footer-text">
    Saudi STT Pipeline — MasriSwitch-Gemma3n — هيئة الزكاة والضريبة والجمارك
</div>
"""

with gr.Blocks(title="Saudi STT — تحويل الصوت إلى نص") as demo:

    gr.HTML(HERO_HTML)

    with gr.Column(elem_classes="card-section"):
        gr.Markdown("### رفع الملف")
        audio_input = gr.Audio(
            label="اسحب ملف صوتي أو فيديو هنا",
            type="filepath",
            sources=["upload", "microphone"],
            elem_classes="audio-input",
        )
        btn = gr.Button(
            "تحويل إلى نص",
            variant="primary",
            size="lg",
        )

    with gr.Column(elem_classes="card-section", visible=True):
        gr.Markdown("### النص المحوّل")
        transcript_output = gr.Textbox(
            label="",
            lines=6,
            rtl=True,
            elem_classes="transcript-box",
            placeholder="النص سيظهر هنا بعد المعالجة...",
        )
        info_bar = gr.Textbox(label="", visible=False)

    with gr.Column(elem_classes="card-section", visible=True):
        gr.Markdown("### إحصائيات")
        stats_output = gr.Markdown(elem_classes="stats-table")

    gr.HTML(FOOTER_HTML)

    btn.click(
        fn=transcribe,
        inputs=[audio_input],
        outputs=[transcript_output, stats_output, info_bar],
    )

if __name__ == "__main__":
    # Preload model
    print("Pre-loading model...")
    load_model()
    print("Starting Gradio server...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True,
                css=ZATCA_CSS)
