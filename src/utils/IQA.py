import warnings
warnings.filterwarnings("ignore")
from typing import List, Tuple, Dict, Optional
import os
from pathlib import Path
import logging
import cv2
from PIL import Image
import numpy as np
import requests
import pyprojroot
import torch
from torchvision import transforms

root = pyprojroot.find_root(pyprojroot.has_dir("src"))


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
    logger.info(f"Computing BRISQUE scores")
    scores = {}
    ext = ["jpg","jpeg","png"]
    for e in ext:
        for img_path in Path(input_dir).glob(f"*.{e}"):
            fname = img_path.name
            # BRISQUE
            gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            try:
                b = brisque_model.compute(gray)
                brisque_score = float(b[0] if isinstance(b, (list,tuple)) else b)
                logger.info(f"BRISQUE score for {fname}: {brisque_score}")
            except Exception as e:
                logger.warning(f"BRISQUE failed for {fname}: {e}")
                brisque_score = None
            scores[fname] = {'brisque': brisque_score}
    if len(scores) > 0:
        logger.info(f"BRISQUE scores computed for {len(scores)} images")
    else:
        logger.warning("No images found in the input directory, check path or no images found in the current directory")

    # classify images into severity levels
    logger.info(f"Classifying images into severity levels")
    scores_by_severity = {
        "very low": {"count":0,"avg_score":0},
        "low": {"count":0,"avg_score":0},
        "medium": {"count":0,"avg_score":0},
        "high": {"count":0,"avg_score":0},
        "very high": {"count":0,"avg_score":0}
    }
    try:
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
            if scores_by_severity[severity]["count"] > 0:
                scores_by_severity[severity]["avg_score"] /= scores_by_severity[severity]["count"]
        
        logger.info(f"Severity levels classified")
        logger.info(f"Severity levels: {scores_by_severity}")
    
    except Exception as e:
        logger.error(f"Error classifying images into severity levels: {e}")
        raise e

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
    if num_images > 0:
        logger.info(f"Metadata generated, Number of images found: {num_images}")
    else:
        logger.warning("No images found in the input directory, check path or no images found in the current directory")

    return stats


class QAlign:
    def __init__(self, host: str = "127.0.0.1", port: int = 5003):
        self.url = f"http://{host}:{port}/qalign_score"

    def query(self, image_path: str) -> float:
        resp = requests.post(self.url, json={"image_path": image_path}, timeout=30)
        resp.raise_for_status()
        return float(resp.json()["score"])

class DepictQA:
    """Parameters when called: img_path_lst, task (eval_degradation or comp_quality), degradations (if task is eval_degradation)."""

    def query(
        self,
        img_path_lst: list[Path],
        task: str,
        degradation: Optional[str] = None,
        replan: bool = False
    ) -> tuple[str, str | list[tuple[str, str]]]:
        assert task in ["eval_degradation", "comp_quality"], f"Unexpected task: {task}"
        if task == "eval_degradation":
            assert (
                len(img_path_lst) == 1
            ), "Only one image should be provided for degradation evaluation."
            return self.eval_degradation(img_path_lst[0], degradation, replan)
        else:
            assert (
                len(img_path_lst) == 2
            ), "Two images should be provided quality comparison."
            return self.compare_img_qual(img_path_lst[0], img_path_lst[1])

    def eval_degradation(
        self, img: Path, degradation: Optional[str], replan: bool = False, previous_plan: Optional[str] = None
    ) -> tuple[str, list[tuple[str, str]]]:
        all_degradations: list[str] = [
            "motion blur",
            "defocus blur",
            "rain",
            "raindrop",
            "haze",
            "dark",
            "noise",
            "jpeg compression artifact",
        ]
        if degradation is None:
            degradations_lst = all_degradations
        else:
            if degradation == "low resolution":
                degradation = "blur"
            else:
                assert isinstance(
                    degradation, str
                ), f"Unexpected type of degradations: {type(degradation)}"
                assert (
                    degradation in all_degradations
                ), f"Unexpected degradation: {degradation}"
            degradations_lst = [degradation]

        levels: set[str] = {"very low", "low", "medium", "high", "very high"}
        res: list[tuple[str, str]] = []
        if replan:
            depictqa_evaluate_degradation_prompt = open(f"{root}/src/prompts/depictqa_eval_replan.md").read()
            logger.info(f"Re-evaluating degradations for {img.name} with prompt: {previous_plan}")
        else:
            depictqa_evaluate_degradation_prompt = open(f"{root}/src/prompts/depictqa_eval.md").read()
            logger.info(f"Evaluating degradation for {img.name} with prompt: {depictqa_evaluate_degradation_prompt}")
        for degradation in degradations_lst:
            if replan:
                prompt = depictqa_evaluate_degradation_prompt.format(
                    degradation=degradation, previous_plan=previous_plan
                )
            else:
                prompt = depictqa_evaluate_degradation_prompt.format(
                    degradation=degradation
                )
            url = "http://127.0.0.1:5001/evaluate_degradation"
            payload = {"imageA_path": img.resolve(), "prompt": prompt}
            rsp: str = requests.post(url, data=payload).json()["answer"]
            assert rsp in levels, f"Unexpected response from DepictQA: {list(rsp)}"
            res.append((degradation, rsp))

        prompt_to_display = depictqa_evaluate_degradation_prompt.format(
            degradation=degradations_lst
        )
        return prompt_to_display, res

    def compare_img_qual(self, img1: Path, img2: Path) -> tuple[str, str]:
        prompt = open(f"{root}/src/prompts/depictqa_compare.md").read()
        url = "http://127.0.0.1:5002/compare_quality"
        payload = {
            "imageA_path": img1.resolve(),
            "imageB_path": img2.resolve(),
            "prompt": prompt
        }
        rsp: str = requests.post(url, data=payload).json()["answer"]

        if "A" in rsp and "B" not in rsp:
            choice = "former"
        elif "B" in rsp and "A" not in rsp:
            choice = "latter"
        else:
            raise ValueError(f"Unexpected answer from DepictQA: {rsp}")

        return prompt, choice

if __name__ == "__main__":

    '''input_path = "/Users/krishnaiyer/generative-ai-agentic-cv-base/data/raw/Test"
    nr_iqa_scores, mean_nr_iqa_scores = nr_iqa(input_path)
    print("NR IQA SCORES")
    print(nr_iqa_scores)
    #results,aggregated = vlm_nr_iqa(input_path, artefacts_path, nr_iqa_scores)

    print("Mean NR IQA SCORES")
    print(mean_nr_iqa_scores)

    print("VLM IQA RESULTS")
    #print(results)

    print("AGGREGATED VLM IQA RESULTS")
    #print(aggregated)'''

    from glob import glob
    depictqa = DepictQA()
    img_path_lst = glob("/home/krishna/workspace/generative-ai-agentic-cv-base/data/raw/*.jpg")
    print(img_path_lst)
    for img_path in img_path_lst:
        prompt_to_display, res = depictqa.query([Path(img_path)], "eval_degradation")
        print(prompt_to_display)
        print(res)
        break

    '''qalign = QAlign()
    print(qalign.query("/home/krishna/workspace/generative-ai-agentic-cv-base/data/raw/foggy-001.jpg"))'''
    