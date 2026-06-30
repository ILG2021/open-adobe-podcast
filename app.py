import os
# Disable Gradio telemetry/analytics
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr
from processor import PIPELINES, process

# ─────────────────────────────────────────────────────────────
# Theme & CSS
# ─────────────────────────────────────────────────────────────

theme = gr.themes.Default(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
).set(
    body_background_fill="*neutral_900",
    body_text_color="white",
    block_background_fill="*neutral_800",
    block_border_color="*neutral_700",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_500",
)

css = """
h1 {
    text-align: center;
    font-weight: 800;
    font-size: 2rem;
    letter-spacing: -0.02em;
    margin-bottom: 0.25rem;
    background: linear-gradient(135deg, #818cf8, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
p.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 0.5rem;
    font-size: 0.95rem;
}
.pipeline-info {
    border-left: 3px solid #6366f1;
    padding: 0.6rem 1rem;
    background: rgba(99,102,241,0.08);
    border-radius: 0 8px 8px 0;
    font-size: 0.85rem;
    color: #cbd5e1;
    margin-top: 0.25rem;
}
.gradio-container {
    max-width: 820px !important;
    margin: auto;
}
footer { display: none !important; }
"""

# ─────────────────────────────────────────────────────────────
# Pipeline descriptions shown beneath the radio buttons
# ─────────────────────────────────────────────────────────────

PIPELINE_DESCRIPTIONS = {
    list(PIPELINES.keys())[0]: (
        "⚡ <b>Two-stage pipeline.</b> "
        "Stage 1: ClearVoice (MossFormer2) aggressively removes noise & reverb. "
        "Stage 2: UniverSR (Flow Matching) reconstructs missing high frequencies up to 48 kHz. "
        "Best for heavily degraded recordings."
    ),
    list(PIPELINES.keys())[1]: (
        "🚀 <b>Single-stage pipeline.</b> "
        "LavaSR (Interspeech 2026) performs noise removal + bandwidth extension in one pass. "
        "~5000× real-time on GPU. Ideal for moderately degraded recordings or near-real-time use."
    ),
}

# ─────────────────────────────────────────────────────────────
# Processing function
# ─────────────────────────────────────────────────────────────

def run_enhancement(audio_path, pipeline_name):
    if audio_path is None:
        return None, "⚠️ Please upload an audio file first."
    if pipeline_name is None:
        return None, "⚠️ Please select a pipeline."
    try:
        output_path = process(audio_path, pipeline_name)
        return output_path, f"✅ Enhancement complete using **{pipeline_name.split('(')[0].strip()}**."
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def update_description(pipeline_name):
    return PIPELINE_DESCRIPTIONS.get(pipeline_name, "")

# ─────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────

pipeline_choices = list(PIPELINES.keys())

with gr.Blocks(theme=theme, css=css, title="Open-Source Podcast AI") as demo:

    gr.Markdown("# 🎙️ Open-Source Podcast AI")
    gr.Markdown("<p class='subtitle'>State-of-the-art speech enhancement — 100% local, 100% private</p>")

    with gr.Row():
        with gr.Column(scale=1):

            # ── Input ──────────────────────────────────────────
            audio_input = gr.Audio(
                type="filepath",
                label="📂 Upload Audio",
                sources=["upload", "microphone"],
            )

            # ── Pipeline Selector ──────────────────────────────
            pipeline_radio = gr.Radio(
                choices=pipeline_choices,
                value=pipeline_choices[0],
                label="🔧 Enhancement Pipeline",
                interactive=True,
            )
            pipeline_desc = gr.HTML(
                value=PIPELINE_DESCRIPTIONS[pipeline_choices[0]],
                elem_classes=["pipeline-info"],
            )
            pipeline_radio.change(
                fn=update_description,
                inputs=pipeline_radio,
                outputs=pipeline_desc,
            )

            # ── Run Button ─────────────────────────────────────
            enhance_btn = gr.Button("✨ Enhance Speech", variant="primary", size="lg")

    with gr.Row():
        with gr.Column(scale=1):
            # ── Output ─────────────────────────────────────────
            audio_output = gr.Audio(
                label="🎧 Enhanced Audio (48 kHz Hi-Fi)",
                interactive=False,
            )
            status_msg = gr.Markdown(value="")

    # ── Event wiring ───────────────────────────────────────────
    enhance_btn.click(
        fn=run_enhancement,
        inputs=[audio_input, pipeline_radio],
        outputs=[audio_output, status_msg],
    )

# ─────────────────────────────────────────────────────────────
# Launch
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
