import argparse
import os

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr

from processor import PIPELINES, match_sound_field, process


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
.gradio-container { max-width: 820px !important; margin: auto; }
footer { display: none !important; }
"""


pipeline_choices = list(PIPELINES)
PIPELINE_DESCRIPTIONS = {
    "🐋 Sidon（多语言语音修复 · 首选）": (
        "<b>首选语音修复方案。</b>Sidon 以最长 96 秒的分块修复语音，输出单声道 "
        "48 kHz 音频，并自动使用官方 CPU 或 CUDA TorchScript 权重。"
    ),
    "🧹 ClearVoice + UniverSR（强力增强 · 按需超分）": (
        "<b>强力语音增强方案。</b>ClearVoice 负责去除底噪和混响；当原始输入为 "
        "8/12/16/24 kHz 时，再由 UniverSR 重建高频。44.1/48 kHz 输入会跳过"
        "不必要的超分辨率，统一输出 48 kHz 音频。"
    ),
    "🌋 LavaSR（极速 · 单阶段）": (
        "<b>单阶段快速增强方案。</b>LavaSR 在一次处理中完成降噪和带宽扩展，"
        "适合对处理速度要求较高的场景。"
    ),
}


def run_enhancement(audio_path, pipeline_name):
    if audio_path is None:
        return None, "⚠️ 请先上传音频文件。"
    if pipeline_name is None:
        return None, "⚠️ 请选择增强方案。"
    try:
        output_path = process(audio_path, pipeline_name)
        return output_path, f"✅ 已使用 **{pipeline_name}** 完成音频增强。"
    except Exception as exc:
        return None, f"❌ 处理失败：{exc}"


def run_matching(target_path, reference_path):
    if target_path is None or reference_path is None:
        return None, "⚠️ 请同时上传待处理音频和参考音频。"
    try:
        output_path = match_sound_field(target_path, reference_path)
        return output_path, "✅ 已使用 **Matchering** 完成声场匹配。"
    except Exception as exc:
        return None, f"❌ 处理失败：{exc}"


def update_description(pipeline_name):
    return PIPELINE_DESCRIPTIONS.get(pipeline_name, "")


with gr.Blocks(title="开源播客音频工作台") as demo:
    gr.Markdown("# 🎙️ 开源播客音频工作台")
    gr.Markdown(
        "<p class='subtitle'>语音增强与参考音频声场匹配 · 完全本地运行 · 保护隐私</p>"
    )

    with gr.Tabs():
        with gr.Tab("✨ 音频增强"):
            enhancement_input = gr.Audio(
                type="filepath",
                label="上传待增强语音",
                sources=["upload", "microphone"],
            )
            pipeline_radio = gr.Radio(
                choices=pipeline_choices,
                value=pipeline_choices[0],
                label="增强方案",
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
            enhancement_button = gr.Button(
                "✨ 开始增强", variant="primary", size="lg"
            )
            enhancement_output = gr.Audio(
                label="增强结果（48 kHz）", interactive=False
            )
            enhancement_status = gr.Markdown(value="")
            enhancement_button.click(
                fn=run_enhancement,
                inputs=[enhancement_input, pipeline_radio],
                outputs=[enhancement_output, enhancement_status],
            )

        with gr.Tab("🎚️ 声场匹配"):
            gr.HTML(
                "<div class='pipeline-info'><b>Matchering 参考匹配。</b>"
                "根据参考音频调整待处理音频的频率响应、RMS 响度、峰值和立体声宽度。"
                "</div>"
            )
            target_input = gr.Audio(
                type="filepath",
                label="待处理音频",
                sources=["upload", "microphone"],
            )
            reference_input = gr.Audio(
                type="filepath",
                label="参考音频",
                sources=["upload"],
            )
            matching_button = gr.Button(
                "🎚️ 开始匹配", variant="primary", size="lg"
            )
            matching_output = gr.Audio(
                label="匹配结果（24-bit WAV）", interactive=False
            )
            matching_status = gr.Markdown(value="")
            matching_button.click(
                fn=run_matching,
                inputs=[target_input, reference_input],
                outputs=[matching_output, matching_status],
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="开源播客音频工作台")
    parser.add_argument(
        "--root_path",
        type=str,
        default=os.environ.get("GRADIO_ROOT_PATH", ""),
        help="Gradio 应用的根路径，例如 /podcast",
    )
    args = parser.parse_args()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        root_path=args.root_path,
        theme=theme,
        css=css,
    )
