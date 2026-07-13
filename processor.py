"""Local audio enhancement and reference-matching processors."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import soundfile as sf
import torch


OUTPUT_DIR = "temp_output"
UNIVERSR_INPUT_SAMPLE_RATES = frozenset({8_000, 12_000, 16_000, 24_000})


def _output_path(input_path: str, prefix: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(input_path).stem
    run_id = uuid4().hex[:12]
    return os.path.join(output_dir, f"{prefix}_{stem}_{run_id}.wav")


def _write_48k_pcm24(input_path: str, output_path: str) -> None:
    """Copy audio to a 48 kHz, 24-bit WAV, resampling only when required."""
    import torchaudio

    audio, sample_rate = sf.read(input_path, always_2d=True, dtype="float32")
    if sample_rate != 48_000:
        tensor = torch.from_numpy(audio.T)
        audio = (
            torchaudio.functional.resample(tensor, sample_rate, 48_000)
            .T.contiguous()
            .numpy()
        )
    sf.write(output_path, audio, 48_000, subtype="PCM_24")


# ---------------------------------------------------------------------------
# Speech enhancement: ClearVoice + UniverSR
# ---------------------------------------------------------------------------

_clearvoice_model = None
_universr_model = None


def _load_clearvoice_universr(device: str):
    global _clearvoice_model, _universr_model
    if _clearvoice_model is None or _universr_model is None:
        print("[ClearVoice+UniverSR] Loading models...")
        from clearvoice import ClearVoice
        from universr import UniverSR

        _clearvoice_model = ClearVoice(
            task="speech_enhancement",
            model_names=["MossFormer2_SE_48K"],
        )
        _universr_model = UniverSR.from_pretrained(
            "woongzip1/universr-audio", device=device
        )
        print("[ClearVoice+UniverSR] Models ready.")
    return _clearvoice_model, _universr_model


def enhance_clearvoice_universr(
    input_path: str, output_dir: str = OUTPUT_DIR
) -> str:
    """Remove noise/reverb, then reconstruct a 48 kHz signal."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_sample_rate = sf.info(input_path).samplerate
    clearvoice_model, universr_model = _load_clearvoice_universr(device)
    denoised_path = _output_path(input_path, "denoised", output_dir)
    final_path = _output_path(input_path, "enhanced_cv_usr", output_dir)

    print("[Stage 1] ClearVoice denoising...")
    denoised_result = clearvoice_model(input_path=input_path, online_write=False)
    clearvoice_model.write(denoised_result, output_path=denoised_path)

    if input_sample_rate in UNIVERSR_INPUT_SAMPLE_RATES:
        print(
            f"[Stage 2] UniverSR super-resolution from {input_sample_rate} Hz..."
        )
        output_tensor = universr_model.enhance(
            denoised_path, input_sr=input_sample_rate
        )
        output = output_tensor.squeeze().detach().cpu().numpy()
        sf.write(final_path, output, 48_000, subtype="PCM_24")
    else:
        print(
            "[Stage 2] Skipping UniverSR: the original input is not one of "
            f"{sorted(UNIVERSR_INPUT_SAMPLE_RATES)} Hz."
        )
        _write_48k_pcm24(denoised_path, final_path)
    return final_path


# ---------------------------------------------------------------------------
# Speech enhancement: LavaSR
# ---------------------------------------------------------------------------

_lavasr_model = None


def _load_lavasr(device: str):
    global _lavasr_model
    if _lavasr_model is None:
        print("[LavaSR] Loading model...")
        from LavaSR.model import LavaEnhance2

        _lavasr_model = LavaEnhance2("YatharthS/LavaSR", device)
        print("[LavaSR] Model ready.")
    return _lavasr_model


def enhance_lavasr(input_path: str, output_dir: str = OUTPUT_DIR) -> str:
    """Run LavaSR's single-stage restoration and output 48 kHz audio."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_lavasr(device)
    final_path = _output_path(input_path, "enhanced_lavasr", output_dir)
    input_sample_rate = sf.info(input_path).samplerate
    input_tensor, _ = model.load_audio(input_path, input_sr=input_sample_rate)
    output_tensor = model.enhance(input_tensor, denoise=True, batch=False)
    output = output_tensor.squeeze().detach().cpu().numpy()
    sf.write(final_path, output, 48_000, subtype="PCM_24")
    return final_path


# ---------------------------------------------------------------------------
# Speech enhancement: Sidon
# Mirrors sarulab-speech/sidon_demo_beta, with CPU fallback and cached models.
# ---------------------------------------------------------------------------

_sidon_models: dict[str, tuple[object, object]] = {}
_sidon_preprocessor = None


def _load_sidon(device: str):
    global _sidon_preprocessor
    if device not in _sidon_models:
        from huggingface_hub import hf_hub_download
        from transformers import SeamlessM4TFeatureExtractor

        suffix = "cuda" if device == "cuda" else "cpu"
        print(f"[Sidon] Loading {suffix.upper()} models...")
        feature_path = hf_hub_download(
            "sarulab-speech/sidon-v0.1",
            filename=f"feature_extractor_{suffix}.pt",
        )
        decoder_path = hf_hub_download(
            "sarulab-speech/sidon-v0.1", filename=f"decoder_{suffix}.pt"
        )
        feature_extractor = torch.jit.load(feature_path, map_location=device).to(device)
        decoder = torch.jit.load(decoder_path, map_location=device).to(device)
        feature_extractor.eval()
        decoder.eval()
        _sidon_models[device] = (feature_extractor, decoder)
        if _sidon_preprocessor is None:
            _sidon_preprocessor = SeamlessM4TFeatureExtractor.from_pretrained(
                "facebook/w2v-bert-2.0"
            )
        print("[Sidon] Models ready.")
    return (*_sidon_models[device], _sidon_preprocessor)


@torch.inference_mode()
def enhance_sidon(input_path: str, output_dir: str = OUTPUT_DIR) -> str:
    """Restore speech with Sidon and write a mono 48 kHz WAV file."""
    import torchaudio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_extractor, decoder, preprocessor = _load_sidon(device)
    waveform, sample_rate = sf.read(input_path, always_2d=True, dtype="float32")
    waveform = waveform.mean(axis=1)
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 0:
        waveform = 0.9 * waveform / peak

    target_samples = round(48_000 / sample_rate * waveform.size)
    audio = torch.from_numpy(waveform).view(1, -1)
    audio = torchaudio.functional.highpass_biquad(audio, sample_rate, 50)
    audio_16k = torchaudio.functional.resample(audio, sample_rate, 16_000)
    audio_16k = torch.nn.functional.pad(audio_16k, (0, 24_000))

    restored_chunks = []
    feature_cache = None
    for chunk in audio_16k.flatten().split(16_000 * 96):
        batch = preprocessor(
            torch.nn.functional.pad(chunk, (160, 160)), return_tensors="pt"
        )
        features = feature_extractor(batch["input_features"].to(device))[
            "last_hidden_state"
        ]
        if feature_cache is not None:
            features = torch.cat((feature_cache, features), dim=1)
        restored_chunks.append(decoder(features.transpose(1, 2)).flatten()[:-960])
        feature_cache = features[:, -1:]

    restored = torch.cat(restored_chunks).detach().cpu().numpy()[:target_samples]
    final_path = _output_path(input_path, "enhanced_sidon", output_dir)
    sf.write(final_path, restored, 48_000, subtype="PCM_24")
    return final_path


# ---------------------------------------------------------------------------
# Sound-field matching: Matchering
# ---------------------------------------------------------------------------


def match_sound_field(
    target_path: str, reference_path: str, output_dir: str = OUTPUT_DIR
) -> str:
    """Master target audio to the reference's spectrum, level and stereo width."""
    import matchering as mg

    final_path = _output_path(target_path, "matched", output_dir)
    mg.process(
        target=target_path,
        reference=reference_path,
        results=[mg.pcm24(final_path)],
    )
    return final_path


PIPELINES = {
    "🐋 Sidon（多语言语音修复 · 首选）": enhance_sidon,
    "🧹 ClearVoice + UniverSR（强力增强 · 按需超分）": enhance_clearvoice_universr,
    "🌋 LavaSR（极速 · 单阶段）": enhance_lavasr,
}


def process(input_path: str, pipeline_name: str) -> str:
    """Route an input file to a selected speech-enhancement pipeline."""
    if pipeline_name not in PIPELINES:
        raise ValueError(f"Unknown pipeline: {pipeline_name}")
    return PIPELINES[pipeline_name](input_path)
