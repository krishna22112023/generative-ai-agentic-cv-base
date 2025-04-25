import os
import json
import logging
from typing import Tuple, Dict

from langchain_core.tools import tool
from .decorators import log_io

from src.utils.IQA import nr_iqa,fr_iqa,vlm_nr_iqa
from src.config import PATHS

logger = logging.getLogger(__name__)

@tool()
@log_io
def no_reference_iqa(prefix:str) -> Tuple[Dict,Dict]:
    """
    Computes no-reference image quality assessment (NR-IQA) using pretrained models,
    without needing a reference image.

    The function relies on the following folder structure under the given prefix:
    - Raw images: PATHS['raw']/{prefix}
    - Artefacts (logs, outputs): PATHS['artefacts']/{prefix}

    The function currently supports two methods:
    
    1. **brisque**:
       - Predicts the BRISQUE score using a support vector regression (SVR) model trained on an image database
         with corresponding differential mean opinion score (DMOS) values.
       - Score range: [0, 100], where 100 indicates the worst quality and 0 is excellent.

    2. **qalign**:
       - Rates image quality on a categorical scale: Excellent, Good, Fair, Poor, Bad.
       - Computes a weighted average of the probabilities for each class.
       - Score range: [1, 5], where 5 is excellent and 1 is bad.

    Based on these metrics, the function uses a VLM to classify images into the following seven degradation types:
    - Noise
    - Motion Blur
    - Defocus Blur
    - Haze
    - Rain
    - Dark
    - JPEG Compression Artifact

    Each degradation is assigned one of five severity levels:
    - Very Low
    - Low
    - Medium
    - High
    - Very High

    Args:
        prefix (str): prefix name in minio bucket

    Returns:
        Tuple: Returns a dictionary of all metrics for each each image and the average score across all images in the input folder.
    """
    input_path = f"{PATHS['raw']}/{prefix}"
    artefacts_path = f"{PATHS['artefacts']}/{prefix}"
    os.makedirs(artefacts_path,exist_ok=True)
    nr_iqa_scores, mean_nr_iqa_scores = nr_iqa(input_path, artefacts_path)
    vlm_nr_iqa_scores, agg_vlm_nr_iqa_scores = vlm_nr_iqa(input_path,artefacts_path,nr_iqa_scores)

    return json.dumps(mean_nr_iqa_scores),json.dumps(agg_vlm_nr_iqa_scores)


@tool()
@log_io
def full_reference_iqa(prefix:str) -> Tuple[Dict,Dict]: 
    """
    Computes full-reference image quality assessment (FR-IQA) scores for images in a given MinIO bucket prefix.
    
    This function compares processed images against their corresponding reference images using standard 
    full-reference IQA metrics. The comparison is done on a per-image basis and supports multiple metrics 
    including PSNR, SSIM, MSE, and GMSD.

    The function relies on the following folder structure under the given prefix:
    - Processed images: PATHS['processed']/{prefix}
    - Reference images: PATHS['raw']/{prefix}
    - Artefacts (logs, outputs): PATHS['artefacts']/{prefix}

    Supported metrics:
    - PSNR (Peak Signal-to-Noise Ratio)
    - SSIM (Structural Similarity Index)
    - MSE (Mean Squared Error)
    - GMSD (Gradient Magnitude Similarity Deviation)

    Args:
        prefix (str): The prefix name used to locate folders in the MinIO bucket.

    Returns:
        Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
            - A nested dictionary mapping each filename to its computed metrics.
            - A dictionary of mean metric scores across all processed images.
    """
    input_path = f"{PATHS['processed']}/{prefix}"
    reference_path = f"{PATHS['raw']}/{prefix}"
    artefacts_path = f"{PATHS['artefacts']}/{prefix}" 
    os.makedirs(artefacts_path,exist_ok=True)
    scores, mean_scores = fr_iqa(input_path, artefacts_path, reference_path)
    return scores,mean_scores

