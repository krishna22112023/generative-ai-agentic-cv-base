import os
import json
import logging
from typing import Tuple, Dict

from langchain_core.tools import tool
from .decorators import log_io

from src.utils.IQA import nr_iqa,fr_iqa,vlm_nr_iqa,get_metadata
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

    # collect metadata
    stats = get_metadata(input_path)
    logger.info("dataset metadata collected")
    with open(os.path.join(artefacts_path,'metadata.json'), 'w') as f:
        json.dump(stats, f, indent=4)

    #compute nr iqa scores
    nr_iqa_scores, mean_nr_iqa_scores = nr_iqa(input_path)
    logger.info("no-reference iqa scores computed")
    with open(os.path.join(artefacts_path, "nr_iqa_results_raw.json"), 'w') as f:
        json.dump(nr_iqa_scores, f, indent=4)

    #compute vlm nr iqa scores
    vlm_nr_iqa_scores, agg_vlm_nr_iqa_scores = vlm_nr_iqa(input_path,artefacts_path,nr_iqa_scores)
    logger.info("vlm no-reference iqa scores computed")
    with open(os.path.join(artefacts_path, "degredation_iqa_results.json"), 'w') as f:
        json.dump(vlm_nr_iqa_scores, f, indent=4)

    return json.dumps(mean_nr_iqa_scores),json.dumps(agg_vlm_nr_iqa_scores)

@tool()
@log_io
def verify_no_reference_iqa(prefix:str) -> list:
    """
    Compares no-reference image quality assessment (NR-IQA) of the processed images with the raw images
    to verify if the preprocessing worked correctly.
    
    The function relies on the following folder structure under the given prefix:
    - Raw images: PATHS['raw']/{prefix}
    - Artefacts (logs, outputs): PATHS['artefacts']/{prefix}

    Args:
        prefix (str): prefix name in minio bucket

    Returns:
        List: List of images that failed the verification, i.e. images that had a negative difference in brisque or qalign
    """
    input_path = f"{PATHS['processed']}/{prefix}"
    artefacts_path = f"{PATHS['artefacts']}/{prefix}"
    os.makedirs(artefacts_path,exist_ok=True)

    # collect previous nr iqa scores from raw data
    with open(f"{artefacts_path}/nr_iqa_results_raw.json", "r") as f:
        nr_iqa_scores_raw = json.load(f)

    # compute processed nr iqa scores from processed data
    nr_iqa_scores_processed, mean_nr_iqa_scores = nr_iqa(input_path, artefacts_path)
    logger.info("no-reference iqa scores computed")
    with open(os.path.join(artefacts_path, "nr_iqa_results_processed.json"), 'w') as f:
        json.dump(nr_iqa_scores_processed, f, indent=4)

    #compare nr iqa scores
    nr_iqa_verification = {}
    verification_failed_images = []
    for fname, nr_iqa_scores in nr_iqa_scores_raw.items():
        brisque_diff = nr_iqa_scores["brisque"] - nr_iqa_scores_processed.get(fname)["brisque"]
        qalign_diff = nr_iqa_scores_processed.get(fname)["qalign"] - nr_iqa_scores["qalign"]
        if brisque_diff <= 0 or qalign_diff <= 0:
            verification_failed_images.append(fname)
        nr_iqa_verification[fname] = {"brisque":brisque_diff,
                                      "qalign":qalign_diff}
    with open(os.path.join(artefacts_path, "nr_iqa_results_verification.json"), 'w') as f:
        json.dump(nr_iqa_verification, f, indent=4)
    
    return verification_failed_images

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

if __name__ == "__main__":
    scores,mean_scores = no_reference_iqa("DAWN/Fog")