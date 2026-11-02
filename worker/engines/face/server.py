"""The face engine on its own (cloud workers): FlashHead live sessions on 127.0.0.1:7012.

On the office machine the same API runs inside the research app's talking-head
process; here it runs alone, with no Gradio and no 2.4 renderer.

Expects SoulX-FlashHead checked out at FLASHHEAD_DIR with its weights under
models/ (docker/fetch-models.sh does both).
"""
import os

# Before torch: torch.compile fails inside inductor on this stack (sympy 1.14),
# and turning it off costs nothing measurable.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("FLASHHEAD_DIR", "/opt/SoulX-FlashHead"))
PORT = int(os.environ.get("AVATAR_PORT", "7012"))

# flash_head/inference.py opens its config by relative path at import time.
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
from flash_head.inference import get_pipeline  # noqa: E402

import avatar_sessions  # noqa: E402

_pipeline = None
_lock = threading.Lock()


def load_pipeline():
    global _pipeline
    if _pipeline is None:
        t0 = time.time()
        print("[Face] loading FlashHead lite…", flush=True)
        _pipeline = get_pipeline(world_size=1, ckpt_dir=str(REPO / "models" / "SoulX-FlashHead-1_3B"),
                                 wav2vec_dir=str(REPO / "models" / "wav2vec2-base-960h"), model_type="lite")
        print(f"[Face] ready in {time.time() - t0:.1f}s on {torch.cuda.get_device_name(0)}", flush=True)
    return _pipeline


if __name__ == "__main__":
    load_pipeline()
    sample = REPO / "examples" / "girl.png"
    # start() runs a self-check render (which also warms the GPU) and serves in a thread.
    if not avatar_sessions.start(PORT, lock=_lock, load_pipeline=load_pipeline,
                                 sample_image=str(sample) if sample.exists() else None):
        sys.exit(1)
    threading.Event().wait()
