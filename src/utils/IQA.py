import warnings
warnings.filterwarnings("ignore")
from typing import List, Tuple, Dict
import os
from pathlib import Path
import logging
import cv2
from PIL import Image
import numpy as np


logger = logging.getLogger(__name__)


def nr_iqa(
    input_dir: str,
    brisque_model: str,
    metrics: List[str] = ['brisque']
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Compute no-reference image quality assessment metrics for all matching images in a directory.

    Args:
        input_path (str): Path to the folder containing the distorted images.
        brisque_model (str): Path to the BRISQUE model.
        metrics (List[str]): List of metrics to compute. Supported: 'brisque'.

    Returns:
        scores (Dict[str, Dict[str, float]]): Nested dict mapping filename -> metric -> score.
        mean_scores (Dict[str, float]): Average score per metric across all processed images.

    Raises:
        ValueError: If no valid images are processed or if any requested metric is unsupported.
    """

    scores = {}
    ext = ["jpg","jpeg","png"]
    for e in ext:
        for img_path in Path(input_dir).glob(e):
            fname = img_path.name
            # BRISQUE
            gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            try:
                b = brisque_model.compute(gray)
                brisque_score = float(b[0] if isinstance(b, (list,tuple)) else b)
            except Exception as e:
                logger.warning(f"BRISQUE failed for {fname}: {e}")
                brisque_score = None

            scores[fname] = {'brisque': brisque_score}

    # classify images into severity levels
    scores_by_severity = {
        "very low": {"count":0,"avg_score":0},
        "low": {"count":0,"avg_score":0},
        "medium": {"count":0,"avg_score":0},
        "high": {"count":0,"avg_score":0},
        "very high": {"count":0,"avg_score":0}
    }
    severity_levels = [20,40,60,80,100]
    for fname, score in scores.items():
        brisque_score = score['brisque']
        if brisque_score is None:
            continue
        elif brisque_score < 20:
            scores_by_severity["very low"]["count"] += 1
            scores_by_severity["very low"]["avg_score"] += brisque_score
        elif brisque_score < 40:
            scores_by_severity["low"]["count"] += 1
            scores_by_severity["low"]["avg_score"] += brisque_score
        elif brisque_score < 60:
            scores_by_severity["medium"]["count"] += 1
            scores_by_severity["medium"]["avg_score"] += brisque_score
        elif brisque_score < 80:
            scores_by_severity["high"]["count"] += 1
            scores_by_severity["high"]["avg_score"] += brisque_score
        else:
            scores_by_severity["very high"]["count"] += 1
            scores_by_severity["very high"]["avg_score"] += brisque_score

    for severity in scores_by_severity:
        scores_by_severity[severity]["avg_score"] /= scores_by_severity[severity]["count"]

    return scores,scores_by_severity


def get_metadata(input_path):
    """
    Scan a directory of images and return statistics in a dictionary:
      - number_of_images
      - size_distribution: counts of images by size category (tiny, small, medium, large, jumbo)
      - aspect_ratio_distribution: counts by aspect ratio category (very_wide, wide, square, tall, extra_wide)
      - average_image_size: (mean width, mean height)
      - median_aspect_ratio

    Size categories (as per Roboflow Health Check UI):
      tiny:    max_dim < 512
      small:   512 <= max_dim < 1024
      medium:  1024 <= max_dim < 2048
      large:   2048 <= max_dim < 4096
      jumbo:   max_dim >= 4096

    Aspect ratio = width / height. Categories:
      extra_wide:     ratio > 2
      very_wide:      1.5 < ratio <= 2
      wide:           1.2 < ratio <= 1.5
      square:         0.8 <= ratio <= 1.2
      tall:           ratio < 0.8
    """
    # thresholds
    size_bins = [512, 1024, 2048, 4096]
    size_labels = ['tiny', 'small', 'medium', 'large', 'jumbo']
    ar_labels = ['extra_wide', 'very_wide', 'wide', 'square', 'tall']

    # initialize counters
    num_images = 0
    size_dist = dict.fromkeys(size_labels, 0)
    ar_dist = dict.fromkeys(ar_labels, 0)
    sizes = []
    aspect_ratios = []

    # scan directory
    for root, _, files in os.walk(input_path):
        for fname in files:
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                num_images += 1
                path = os.path.join(root, fname)
                try:
                    with Image.open(path) as img:
                        w, h = img.size
                except Exception:
                    continue

                # record
                sizes.append((w, h))
                ar = w / h if h else 0
                aspect_ratios.append(ar)

                # size category by max dimension
                max_dim = max(w, h)
                for i, thresh in enumerate(size_bins):
                    if max_dim < thresh:
                        size_dist[size_labels[i]] += 1
                        break
                else:
                    size_dist['jumbo'] += 1

                # aspect ratio category
                if ar > 2:
                    ar_dist['extra_wide'] += 1
                elif ar > 1.5:
                    ar_dist['very_wide'] += 1
                elif ar > 1.2:
                    ar_dist['wide'] += 1
                elif ar >= 0.8:
                    ar_dist['square'] += 1
                else:
                    ar_dist['tall'] += 1

    # compute avg and median
    if sizes:
        widths, heights = zip(*sizes)
        avg_w, avg_h = float(np.mean(widths)), float(np.mean(heights))
    else:
        avg_w = avg_h = 0.0

    median_ar = float(np.median(aspect_ratios)) if aspect_ratios else 0.0

    stats = {
        'number_of_images': num_images,
        'size_distribution': size_dist,
        'aspect_ratio_distribution': ar_dist,
        'average_image_size': {'width': avg_w, 'height': avg_h},
        'median_aspect_ratio': median_ar
    }

    return stats


if __name__ == "__main__":

    input_path = "/Users/krishnaiyer/generative-ai-agentic-cv-base/data/raw/Test"

    nr_iqa_scores, mean_nr_iqa_scores = nr_iqa(input_path)
    print("NR IQA SCORES")
    print(nr_iqa_scores)
    #results,aggregated = vlm_nr_iqa(input_path, artefacts_path, nr_iqa_scores)

    print("Mean NR IQA SCORES")
    print(mean_nr_iqa_scores)

    print("VLM IQA RESULTS")
    #print(results)

    print("AGGREGATED VLM IQA RESULTS")
    #print(aggregated)
