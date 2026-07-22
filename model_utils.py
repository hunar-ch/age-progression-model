"""
Core inference logic for the age progression/regression GUI.

Key design point: the network's forward pass is expensive; the alpha-scaling
of the aging residual is cheap. So we run the model ONCE per uploaded image,
cache the residual, and every alpha (-1.5 .. +1.5) is just numpy math on that
cached residual. This is what makes the GUI feel instant when switching
between the six age-shift steps.
"""

import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models import Generator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__), "pretrained_model", "state_dict.pth"
)

# UI label -> alpha. Matches the plan doc's mapping.
ALPHA_MAP = {
    "-30": -1.5,
    "-20": -1.0,
    "-10": -0.5,
    "+10": 0.5,
    "+20": 1.0,
    "+30": 1.5,
}

_device = None
_model = None

_transform = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)

# Very small in-memory cache: last uploaded image's orig_np + residual.
# Keyed by a cheap hash of the image bytes so re-selecting the same image
# doesn't re-run the network either.
_cache = {"key": None, "orig_np": None, "residual": None}


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"[model_utils] Using device: {_device}")
    return _device


def load_model() -> torch.nn.Module:
    """Load the Generator once and keep it in eval mode on the target device."""
    global _model
    if _model is not None:
        return _model

    device = get_device()
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. "
            "Place your state_dict.pth in pretrained_model/."
        )

    model = Generator(ngf=32, n_residual_blocks=9).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    _model = model
    print("[model_utils] Model loaded and set to eval mode.")
    return _model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _image_key(image: Image.Image) -> str:
    """Cheap cache key: image size + a hash of a downsampled thumbnail."""
    thumb = image.resize((32, 32)).convert("RGB")
    return f"{image.size}-{hash(thumb.tobytes())}"


def _denormalize(tensor_np: np.ndarray) -> np.ndarray:
    """
    Sign-safe denormalization. If values are negative, treat the tensor as
    normalized to [-1, 1] (our transform's convention) and rescale to [0, 1].
    Otherwise assume it's already in [0, 1] range.
    """
    if tensor_np.min() < 0.0:
        return np.clip((tensor_np + 1.0) / 2.0, 0.0, 1.0)
    return np.clip(tensor_np, 0.0, 1.0)


def _compute_residual(image: Image.Image):
    """Run the model once, return (orig_np, residual) in [0, 1] space."""
    device = get_device()
    model = load_model()

    img_tensor = _transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        aged_tensor = model(img_tensor)

    t_orig = img_tensor.squeeze().detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
    t_aged = aged_tensor.squeeze().detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)

    orig_np = _denormalize(t_orig)
    aged_np = _denormalize(t_aged)
    residual = aged_np - orig_np

    return orig_np, residual


def generate(image: Image.Image, alpha: float) -> Image.Image:
    """
    Apply the cached aging residual scaled by `alpha` to `image`.
    Runs the network only if the image isn't already cached.
    """
    key = _image_key(image)

    if _cache["key"] != key:
        orig_np, residual = _compute_residual(image)
        _cache["key"] = key
        _cache["orig_np"] = orig_np
        _cache["residual"] = residual
    else:
        orig_np = _cache["orig_np"]
        residual = _cache["residual"]

    current_np = np.clip(orig_np + alpha * residual, 0.0, 1.0)
    current_uint8 = (current_np * 255).astype(np.uint8)
    return Image.fromarray(current_uint8)


def generate_all_steps(image: Image.Image) -> dict:
    """Convenience helper for the gallery view: returns {label: PIL.Image} for all 6 steps."""
    return {label: generate(image, alpha) for label, alpha in ALPHA_MAP.items()}