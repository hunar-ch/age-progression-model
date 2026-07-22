"""
Optional metrics: PSNR, SSIM, and FaceNet identity similarity between the
original image and a generated result.

FaceNet is loaded lazily (only on first use) so the app starts fast and
doesn't pull in facenet-pytorch / do the extra forward pass unless the
metrics panel is actually toggled on.
"""

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

from model_utils import get_device

_facenet = None


def _load_facenet():
    global _facenet
    if _facenet is None:
        from facenet_pytorch import InceptionResnetV1
        device = get_device()
        _facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        print("[metrics_utils] FaceNet loaded.")
    return _facenet


def _to_facenet_tensor(image_np: np.ndarray) -> torch.Tensor:
    """image_np in [0, 1], HWC -> normalized [-1, 1] CHW tensor on device."""
    device = get_device()
    tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
    tensor = torch.clamp((tensor * 2.0) - 1.0, -1.0, 1.0)
    return tensor


def compute_metrics(orig_np: np.ndarray, current_np: np.ndarray, alpha: float) -> dict:
    """
    orig_np, current_np: HWC arrays in [0, 1], same shape.
    Returns PSNR, SSIM, and FaceNet cosine identity similarity.
    """
    facenet = _load_facenet()

    ssim_val = ssim_metric(orig_np, current_np, data_range=1.0, channel_axis=2)
    psnr_val = (
        psnr_metric(orig_np, current_np, data_range=1.0) if alpha != 0 else float("inf")
    )

    with torch.no_grad():
        emb_orig = facenet(_to_facenet_tensor(orig_np)).squeeze().cpu().numpy()
        emb_current = facenet(_to_facenet_tensor(current_np)).squeeze().cpu().numpy()

    cos_sim = float(
        np.dot(emb_orig, emb_current)
        / (np.linalg.norm(emb_orig) * np.linalg.norm(emb_current))
    )

    return {
        "psnr": psnr_val,
        "ssim": float(ssim_val),
        "identity_similarity": cos_sim,
    }


def format_metrics_markdown(metrics: dict) -> str:
    psnr_str = f"{metrics['psnr']:.2f} dB" if metrics["psnr"] != float("inf") else "Baseline (no change)"
    return (
        f"**PSNR:** {psnr_str}  \n"
        f"**SSIM:** {metrics['ssim']:.4f}  \n"
        f"**FaceNet identity similarity:** {metrics['identity_similarity']:.4f}"
    )