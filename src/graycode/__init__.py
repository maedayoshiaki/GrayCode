"""graycode — projector–camera gray-code geometric calibration and image warping.

Public reuse surface for downstream projects (e.g. radiometric compensation
loops that drive their own camera). Camera capture is injectable via the
``capture_fn`` argument of :func:`run_capture` / :func:`run_calibration`; the
Canon EDSDK default in :func:`capture` is imported lazily so this package
imports without the Canon SDK.
"""

from . import coords
from .cap_graycode import capture, run_calibration, run_capture
from .decode import decode_c2p
from .gen_graycode import generate_expanded_patterns, save_patterns
from .interpolate_c2p import load_c2p_numpy_array
from .interpolate_p2c import load_p2c_numpy_array
from .warp_image import PixelMapWarperTorch

__all__ = [
    "PixelMapWarperTorch",
    "capture",
    "coords",
    "decode_c2p",
    "generate_expanded_patterns",
    "load_c2p_numpy_array",
    "load_p2c_numpy_array",
    "run_calibration",
    "run_capture",
    "save_patterns",
]
