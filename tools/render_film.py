"""Render one face + one voice into an MP4 with FlashHead Lite or Pro, offline.

Not the live gateway: this loads its own copy of the model in a separate
process, so it needs free GPU memory next to whatever else runs on the card.
It checks that first and stops with a clear message instead of crowding out
the live engine.

Run it with the FlashHead venv (it has torch and the model code):

    ~/age-progression-app/face-flashhead/venv310/bin/python tools/render_film.py \
        --model pro --image amara.jpg --audio amara_en.mp3 --out amara_en_pro.mp4 [--resolution 512]

Then compare two renders with tools/compare_film.sh.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

FLASHHEAD = Path(os.environ.get("FLASHHEAD_DIR", Path.home() / "age-progression-app/face-flashhead/SoulX-FlashHead"))
CKPT = FLASHHEAD / "models/SoulX-FlashHead-1_3B"
WAV2VEC = FLASHHEAD / "models/wav2vec2-base-960h"
# Measured need is about 15 GB for Lite; Pro uses the same 1.3B size plus the
# larger Wan VAE. Keep a margin so the live engine is never pushed out of memory.
MIN_FREE_GB = {"lite": 16, "pro": 18}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["lite", "pro"], required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--audio", required=True, help="any format ffmpeg reads (mp3, wav, ...)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resolution", type=int, default=512, help="square size in px, a multiple of 32 (default 512)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-crop", action="store_true", help="skip face detection and framing")
    ap.add_argument("--force", action="store_true", help="skip the free GPU memory check")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.resolution % 32:
        sys.exit("--resolution must be a multiple of 32")
    image, audio, out = (str(Path(p).resolve()) for p in (args.image, args.audio, args.out))

    import torch
    free_gb = torch.cuda.mem_get_info()[0] / 2**30
    need = MIN_FREE_GB[args.model]
    print(f"GPU free: {free_gb:.1f} GB (need about {need} GB for {args.model})", flush=True)
    if free_gb < need and not args.force:
        sys.exit(f"Not enough free GPU memory for {args.model}: {free_gb:.1f} GB free. "
                 "Nothing was stopped; run again when the card has room (or --force).")

    # The model code reads its config relative to its own folder.
    os.chdir(FLASHHEAD)
    sys.path.insert(0, str(FLASHHEAD))
    import librosa
    import numpy as np
    from flash_head import inference as fh

    t0 = time.time()
    pipeline = fh.get_pipeline(world_size=1, ckpt_dir=str(CKPT), model_type=args.model, wav2vec_dir=str(WAV2VEC))
    fh.infer_params["height"] = fh.infer_params["width"] = args.resolution
    fh.get_base_data(pipeline, cond_image_path_or_dir=image, base_seed=args.seed, use_face_crop=not args.no_crop)
    p = fh.get_infer_params()
    t_load = time.time()

    sr, fps = p["sample_rate"], p["tgt_fps"]
    motion = p["motion_frames_num"]
    slice_len = p["frame_num"] - motion  # new frames per chunk: 24 for Lite, 28 for Pro
    slice_samples = slice_len * sr // fps
    speech, _ = librosa.load(audio, sr=sr, mono=True)
    spoken_sec = len(speech) / sr
    speech = np.pad(speech, (0, (-len(speech)) % slice_samples)).astype(np.float32)

    # Raw frames straight into ffmpeg, voice muxed in, video cut to the voice's length.
    size = args.resolution
    enc = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size}x{size}", "-r", str(fps), "-i", "pipe:0",
        "-i", audio, "-map", "0:v:0", "-map", "1:a:0",
        "-vf", "format=yuv420p", "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-bf", "0",
        "-c:a", "aac", "-b:a", "160k", "-cutoff", "20000",
        "-t", f"{spoken_sec:.3f}", "-movflags", "+faststart", out,
    ], stdin=subprocess.PIPE)

    window_len = sr * p["cached_audio_duration"]
    end_idx = p["cached_audio_duration"] * fps
    window = deque([0.0] * window_len, maxlen=window_len)
    frames = 0
    torch.cuda.synchronize()
    t_render0 = time.time()
    for chunk in speech.reshape(-1, slice_samples):
        window.extend(chunk.tolist())
        emb = fh.get_audio_embedding(pipeline, np.array(window, dtype=np.float32), end_idx - p["frame_num"], end_idx)
        video = fh.run_pipeline(pipeline, emb)[motion:]
        enc.stdin.write(video.to(torch.uint8).cpu().numpy().tobytes())
        frames += video.shape[0]
    torch.cuda.synchronize()
    t_render = time.time() - t_render0
    enc.stdin.close()
    if enc.wait() != 0:
        sys.exit("ffmpeg failed")

    print(f"{Path(out).name}: model {args.model}, {size}x{size}, {spoken_sec:.1f} s of voice, {frames} frames")
    print(f"load {t_load - t0:.1f} s, render {t_render:.1f} s, {frames / t_render:.1f} FPS "
          f"({frames / t_render / fps:.2f}x real time), peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")


if __name__ == "__main__":
    main()
