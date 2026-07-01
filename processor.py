"""
Audio Enhancement Processor
Supports two pipelines:
  - "clearvoice_universr": ClearVoice (MossFormer2) denoising → UniverSR super-resolution
  - "lavasr": LavaSR single-stage speech restoration & bandwidth extension
"""

import os
import torch
import soundfile as sf

# ─────────────────────────────────────────
# Pipeline 1: ClearVoice + UniverSR
# ─────────────────────────────────────────

_clearvoice_model = None
_universr_model = None

def _load_clearvoice_universr(device):
    global _clearvoice_model, _universr_model
    if _clearvoice_model is None or _universr_model is None:
        print("[ClearVoice+UniverSR] Loading models...")
        from clearvoice import ClearVoice
        from universr import UniverSR
        _clearvoice_model = ClearVoice(
            task='speech_enhancement',
            model_names=['MossFormer2_SE_48K']
        )
        _universr_model = UniverSR.from_pretrained(
            "woongzip1/universr-audio",
            device=device
        )
        print("[ClearVoice+UniverSR] Models ready.")
    return _clearvoice_model, _universr_model


def enhance_clearvoice_universr(input_path: str, output_dir: str = "temp_output") -> str:
    """
    Two-stage pipeline:
      Stage 1 – ClearVoice (MossFormer2) removes noise / reverb.
      Stage 2 – UniverSR (Flow Matching) upsamples to 48 kHz hi-fi audio.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clearvoice_model, universr_model = _load_clearvoice_universr(device)

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    denoised_path = os.path.join(output_dir, f"denoised_{base_name}.wav")
    final_path    = os.path.join(output_dir, f"enhanced_cv_usr_{base_name}.wav")

    # Stage 1 – Denoise
    print("[Stage 1] ClearVoice denoising...")
    denoised_result = clearvoice_model(input_path=input_path, online_write=False)
    clearvoice_model.write(denoised_result, output_path=denoised_path)

    # Stage 2 – Super-resolution
    print("[Stage 2] UniverSR super-resolution...")
    # Get sample rate using soundfile
    info = sf.info(denoised_path)
    sr = info.samplerate
    output_tensor = universr_model.enhance(denoised_path, input_sr=sr)
    
    # Save output using soundfile
    output_np = output_tensor.squeeze().cpu().numpy()
    if output_np.ndim == 1:
        output_np = output_np.reshape(-1, 1) # Ensure correct shape for writing
    sf.write(final_path, output_np, 48000)

    print(f"[Done] Saved → {final_path}")
    return final_path


# ─────────────────────────────────────────
# Pipeline 2: LavaSR
# ─────────────────────────────────────────

_lavasr_model = None

def _load_lavasr(device):
    global _lavasr_model
    if _lavasr_model is None:
        print("[LavaSR] Loading model...")
        # Following official HuggingFace Space integration
        from LavaSR.model import LavaEnhance2
        _lavasr_model = LavaEnhance2("YatharthS/LavaSR", device)
        print("[LavaSR] Model ready.")
    return _lavasr_model


def enhance_lavasr(input_path: str, output_dir: str = "temp_output") -> str:
    """
    Single-stage pipeline using LavaSR (Interspeech 2026).
    ~5000x real-time on GPU, ~60x on CPU. Outputs 48 kHz audio.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_lavasr(device)

    os.makedirs(output_dir, exist_ok=True)
    base_name  = os.path.splitext(os.path.basename(input_path))[0]
    final_path = os.path.join(output_dir, f"enhanced_lavasr_{base_name}.wav")

    print("[LavaSR] Enhancing audio...")
    
    # 1. Get exact sample rate first (using soundfile)
    info = sf.info(input_path)
    input_sr = info.samplerate

    # 2. Load audio using LavaSR's native method
    input_audio_tensor, actual_sr = model.load_audio(input_path, input_sr=input_sr)
    
    # 3. Enhance
    output_audio_tensor = model.enhance(
        input_audio_tensor, 
        denoise=True, 
        batch=False
    )
    
    # Save the output (it outputs 48kHz by default) using soundfile
    output_np = output_audio_tensor.squeeze().cpu().numpy()
    if output_np.ndim == 1:
        output_np = output_np.reshape(-1, 1) # Ensure correct shape for writing
    sf.write(final_path, output_np, 48000)

    print(f"[Done] Saved → {final_path}")
    return final_path


# ─────────────────────────────────────────
# Unified entry point
# ─────────────────────────────────────────

PIPELINES = {
    "🧹 ClearVoice + UniverSR  (Max Quality – Two-Stage)": enhance_clearvoice_universr,
    "🌋 LavaSR  (Ultra-Fast – Single-Stage)":               enhance_lavasr,
}

def process(input_path: str, pipeline_name: str) -> str:
    """
    Route audio to the selected enhancement pipeline.
    Returns the path to the output file.
    """
    if pipeline_name not in PIPELINES:
        raise ValueError(f"Unknown pipeline: {pipeline_name}")
    return PIPELINES[pipeline_name](input_path)
