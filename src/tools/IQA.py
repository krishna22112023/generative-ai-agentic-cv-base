import os
import json
from pathlib import Path
import shutil
import logging
from typing import Tuple, Dict
import torch
import cv2
from transformers import AutoModelForCausalLM

from langchain_core.tools import tool
from .decorators import log_io

from src.utils.IQA import nr_iqa,fr_iqa,vlm_nr_iqa,get_metadata

root = pyprojroot.find_root(pyprojroot.has_dir("src"))
DATA_DIR = os.getenv("DATA_DIR")

logger = logging.getLogger(__name__)

@tool()
@log_io
def no_reference_iqa() -> Tuple[Dict,Dict]:
    """
    Computes no-reference image quality assessment (NR-IQA) using pretrained models,
    without needing a reference image.

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

    Returns:
        Tuple: Returns a dictionary of all metrics for each each image and the average score across all images in the input folder.
    """
    input_path = f"{DATA_DIR}/raw"
    artefacts_path = f"{DATA_DIR}/artefacts"
    os.makedirs(artefacts_path,exist_ok=True)

    # collect metadata
    stats = get_metadata(input_path)
    logger.info("dataset metadata collected")
    with open(os.path.join(artefacts_path,'metadata.json'), 'w') as f:
        json.dump(stats, f, indent=4)

    #compute nr iqa scores
    #load models
    qalign_model = AutoModelForCausalLM.from_pretrained("q-future/one-align", trust_remote_code=True, attn_implementation="eager", 
                                        torch_dtype=torch.float16, device_map="auto")
    brisque_model = cv2.quality.QualityBRISQUE_create("src/config/brisque/brisque_model_live.yml","src/config/brisque/brisque_range_live.yml")
    nr_iqa_scores, mean_nr_iqa_scores = nr_iqa(input_path,qalign_model, brisque_model)
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
def verify_no_reference_iqa() -> list:
    """
    For each image under processed/<prefix>/<image>/models/<model>.jpg:
      1) load raw BRISQUE/QAlign from nr_iqa_results_raw.json
      2) compute per-model scores via nr_iqa()
      3) for each image, pick model maximizing (qalign_processed - qalign_raw)
      4) save the chosen image to processed_final/<prefix>/
      5) collect per-image & per-model metrics and diffs into JSON

    Returns list of images where no model improved both metrics.
    """
    # paths
    raw_scores_path     = os.path.join(DATA_DIR,'artefacts', 'nr_iqa_results_raw.json')
    proc_root           = os.path.join(DATA_DIR,'processed')
    final_root          = os.path.join(DATA_DIR,'processed_final')
    metrics_out_path    = os.path.join(DATA_DIR,'artefacts', 'preprocess_comparison_metrics.json')
    os.makedirs(final_root, exist_ok=True)
    os.makedirs(os.path.dirname(metrics_out_path), exist_ok=True)

    # load raw
    with open(raw_scores_path, 'r') as f:
        raw_scores = json.load(f)

    #load models
    qalign_model = AutoModelForCausalLM.from_pretrained("q-future/one-align", trust_remote_code=True, attn_implementation="eager", 
                                        torch_dtype=torch.float16, device_map="auto")
    brisque_model = cv2.quality.QualityBRISQUE_create("src/config/brisque/brisque_model_live.yml","src/config/brisque/brisque_range_live.yml")

    comparison: Dict[str, Dict] = {}
    failures: list = []

    # iterate each image folder
    for img_folder in os.listdir(proc_root):
        img_name = img_folder + ".jpg"  # e.g. "foggy-004.jpg" → folder named "foggy-004"
        raw = raw_scores.get(img_name)
        if not raw:
            logger.warning(f"No raw IQA for {img_name}, skipping")
            continue

        # gather per-model
        model_dirs = Path(proc_root, img_folder, 'models')
        per_model_scores = {}
        for mdl in os.listdir(model_dirs):
            mdl_path = os.path.join(model_dirs,mdl,img_name)
            if not os.path.exists(mdl_path):
                logger.warning(f"Missing output for {mdl_path}")
                continue
            scores, _ = nr_iqa(str(model_dirs / mdl), qalign_model, brisque_model)
            # nr_iqa returns a dict, but we only care about this one file
            per_model_scores[mdl] = scores.get(img_name, {})

        # compute diffs
        best_model = None
        best_diff  = float('-inf')
        metrics_record = {}
        for mdl, sc in per_model_scores.items():
            br_diff = raw['brisque'] - sc.get('brisque', raw['brisque'])
            qa_diff = sc.get('qalign', 0) - raw['qalign']
            metrics_record[mdl] = {
                'brisque_raw': raw['brisque'],
                'brisque_proc': sc.get('brisque'),
                'brisque_diff': br_diff,
                'qalign_raw': raw['qalign'],
                'qalign_proc': sc.get('qalign'),
                'qalign_diff': qa_diff
            }
            # choose by max qalign_diff
            if qa_diff > best_diff:
                best_diff = qa_diff
                best_model = mdl

        comparison[img_name] = {
            'chosen_model': best_model,
            'models': metrics_record
        }

        # copy best output
        if best_model:
            src = model_dirs / best_model / img_name
            dst = Path(final_root) / img_name
            shutil.copy2(src, dst)
        else:
            failures.append(img_name)

    # write out JSON
    with open(metrics_out_path, 'w') as f:
        json.dump(comparison, f, indent=4)

    return comparison


if __name__ == "__main__":
    comparison = verify_no_reference_iqa("Test")