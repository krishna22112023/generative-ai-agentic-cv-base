import cv2
import numpy as np
from typing import Tuple, Union

__all__ = [
    "blur",
    "gaussian_blur",
    "median_blur",
    "bilateral_filter",
    "fast_nl_means_denoising_colored",
    "equalize_hist",
    "clahe_equalize",
    "convert_scale_abs",
    "gamma_correction"
]

# -----------------------------------------------------------------------------
# Basic blurring / smoothing operations
# -----------------------------------------------------------------------------

def blur(image: np.ndarray, ksize: Tuple[int, int] = (5, 5)) -> np.ndarray:
    """Wrapper around ``cv2.blur``.

    Parameters
    ----------
    image : np.ndarray
        Input BGR/Gray image.
    ksize : Tuple[int, int]
        Kernel size (width, height).

    Returns
    -------
    np.ndarray
        Blurred image.
    """
    return cv2.blur(image, ksize)


def gaussian_blur(
    image: np.ndarray,
    ksize: Tuple[int, int] = (5, 5),
    sigmaX: float = 0,
    sigmaY: float = 0,
) -> np.ndarray:
    """Gaussian blur wrapper.

    See ``cv2.GaussianBlur`` for parameter descriptions.
    """
    return cv2.GaussianBlur(image, ksize, sigmaX, sigmaY)


def median_blur(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Median blur wrapper.

    The kernel size **must** be an odd integer > 1.
    """
    return cv2.medianBlur(image, ksize)


def bilateral_filter(
    image: np.ndarray,
    d: int = 9,
    sigmaColor: float = 75,
    sigmaSpace: float = 75,
) -> np.ndarray:
    """Bilateral filtering.

    Parameters mirror ``cv2.bilateralFilter``.
    """
    return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)


def fast_nl_means_denoising_colored(
    image: np.ndarray,
    h: float = 10.0,
    hColor: float = 10.0,
    templateWindowSize: int = 7,
    searchWindowSize: int = 21,
) -> np.ndarray:
    """Colored version of Fast Non-Local Means denoising wrapper."""
    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        h,
        hColor,
        templateWindowSize,
        searchWindowSize,
    )

# -----------------------------------------------------------------------------
# Intensity equalisation (require single-channel operations)
# -----------------------------------------------------------------------------

def _apply_single_channel(function, image: np.ndarray) -> np.ndarray:
    """Utility to apply a single-channel function to colour images.

    The function is applied on the *luminance* channel (Y in YCrCb space),
    then the image is converted back to BGR.
    """
    if len(image.shape) == 2 or image.shape[2] == 1:  # Grayscale
        return function(image)
    # Assume BGR input
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = function(y)
    ycrcb_eq = cv2.merge((y_eq, cr, cb))
    return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)


def equalize_hist(image: np.ndarray) -> np.ndarray:
    """Histogram equalisation using ``cv2.equalizeHist``.

    For colour images, the equalisation is performed on the luminance channel
    (Y) in YCrCb space to avoid introducing colour artefacts.
    """
    return _apply_single_channel(cv2.equalizeHist, image)


def clahe_equalize(
    image: np.ndarray,
    clipLimit: float = 2.0,
    tileGridSize: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalisation (CLAHE).

    Similar colour handling strategy as :func:`equalize_hist`.
    """
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return _apply_single_channel(clahe.apply, image)

# -----------------------------------------------------------------------------
# Intensity / tone mapping helpers
# -----------------------------------------------------------------------------

def convert_scale_abs(
    image: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> np.ndarray:
    """Scale, bias then take absolute value and convert to uint8 via OpenCV."""
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Apply gamma correction.

    ``output = input ** (1 / gamma)`` after normalising to 
    the range [0, 1].
    """
    if gamma <= 0:
        raise ValueError("Gamma must be > 0")
    inv_gamma = 1.0 / gamma
    # Build lookup table
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(
        "uint8"
    )
    return cv2.LUT(image, table)


