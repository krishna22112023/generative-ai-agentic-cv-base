"""
The following are helper function for image preprocessing using : 
1. OpenCV : functions include blur, gaussian_blur, median_blur, bilateral_filter, fast_nl_means_denoising_colored, equalize_hist, clahe_equalize, convert_scale_abs, gamma_correction
2. qwen-image-edit : api function call to the qwen-image-edit model using dashscope library 
"""
import cv2
import numpy as np
from typing import Tuple, Union
import json
import os
from time import time
from PIL import Image
import dashscope
import logging
import requests
import pyprojroot
root = pyprojroot.find_root(pyprojroot.has_dir("src"))
from dashscope import MultiModalConversation

logger = logging.getLogger(__name__)

dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'

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


def resize_image(img_response,height,width):
    try:
        import torch
        import torch.nn.functional as F
        import torchvision.transforms as T
        from io import BytesIO
        import os

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert the response bytes to a PIL image
        response_pil = Image.open(BytesIO(img_response.content)).convert("RGB")
        tensor_img = T.ToTensor()(response_pil).unsqueeze(0).to(device)

        # Resize tensor to original (height, width)
        resized_tensor = F.interpolate(tensor_img, size=(height, width), mode="bilinear", align_corners=False)

        # Convert back to PIL and save
        resized_pil = T.ToPILImage()(resized_tensor.squeeze(0).cpu())
        return resized_pil
    except Exception as e:
        logger.error(f"Failed to resize with torch: {e}. Saving original image.")
        return img_response.content


def qwen_preprocess(image_path: str, degradation_severity:Tuple[str,str], output_path: str) -> str:

    prompt = open(f"{root}/src/prompts/qwen_preprocessor.md").read()
    formatted_prompt = prompt.format(degradation_type=degradation_severity[0], severity=degradation_severity[1])
    print("Prompt: ", formatted_prompt)

    messages = [
        {
            "role": "user",
            "content": [
                {"image": image_path},
                {"text": formatted_prompt}
            ]
        }
    ]

    # If you have not configured an environment variable, replace the following line with your Model Studio API key: api_key="sk-xxx"
    api_key = os.getenv("DASHSCOPE_API_KEY")

    start_time = time()
    response = MultiModalConversation.call(
        api_key=api_key,
        model="qwen-image-edit",
        messages=messages,
        result_format='message',
        stream=False,
        negative_prompt=""
    )
    print("Response: ", response)
    end_time = time()
    print(f"inference time : {end_time - start_time} seconds")
    if response.status_code == 200:
        print("Qwen image process complete!")
        content = response.output["choices"][0]["message"]["content"][0]["image"]
        img_response = requests.get(content)
        # Get original image resolution
        image = cv2.imread(image_path)
        if image is not None:
            height, width, channels = image.shape
            print(f"Original image resolution: {width}x{height} ({channels} channels)")
        else:
            # Fallback to PIL if cv2 fails
            with Image.open(image_path) as img:
                width, height = img.size
                print(f"Original image resolution: {width}x{height}")
        
        # Resize the processed image to match the original resolution using Torch with CUDA acceleration
        resized_pil = resize_image(img_response, height, width)
        save_path = os.path.join(
            output_path,
            f"{os.path.basename(image_path)}-{degradation_severity[0]}-{degradation_severity[1]}.jpg"
        )
        resized_pil.save(save_path, format="JPEG")
        print(f"Image saved to {output_path}")
        return response
    else:
        logger.error(f"HTTP status code: {response.status_code}")
        logger.error(f"Error code: {response.code}")
        logger.error(f"Error message: {response.message}")

if __name__ == "__main__":
    image_path = "/home/krishna/workspace/generative-ai-agentic-cv-base/data/raw/foggy-001.jpg"
    degradation_severity = ("haze", "high")
    output_path = "/home/krishna/workspace/generative-ai-agentic-cv-base/data/raw/"
    qwen_preprocess(image_path, degradation_severity, output_path)