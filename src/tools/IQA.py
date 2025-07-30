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

from src.utils.IQA import nr_iqa,get_metadata
from src.utils.minio import Create

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

    Each degradation is assigned one of five severity levels based on the BRISQUE score:
    0-20 → “very low” 
    20-40 → “low” 
    40-60 → “medium”
    60-80 → “high” 
    80-100 → “very high” 

    Returns:
        Tuple: Return a dictionary of the number of images and the average score in each degradation severity level
    """
    DATA_DIR = os.getenv("DATA_DIR")
    PROJECT_NAME = os.getenv("PROJECT_NAME")
    input_path = f"{DATA_DIR}/raw"
    artefacts_path = f"{DATA_DIR}/artefacts"
    os.makedirs(artefacts_path,exist_ok=True)

    # collect metadata
    stats = get_metadata(input_path)
    with open(os.path.join(artefacts_path,'metadata.json'), 'w') as f:
        json.dump(stats, f, indent=4)

    #load models
    brisque_model = cv2.quality.QualityBRISQUE_create("src/config/brisque/brisque_model_live.yml","src/config/brisque/brisque_range_live.yml")
    scores,scores_by_severity = nr_iqa(input_path, brisque_model)
    
    with open(os.path.join(artefacts_path, "nr_iqa_results_raw.json"), 'w') as f:
        json.dump(scores, f, indent=4)
    with open(os.path.join(artefacts_path, "nr_iqa_results_by_raw_severity.json"), 'w') as f:
        json.dump(scores_by_severity, f, indent=4)

    create = Create()
    create.upload_object(artefacts_path, f"{PROJECT_NAME}/artefacts")
    logger.info(f"IQA results uploaded to {PROJECT_NAME}/artefacts in minio")

    return json.dumps(scores_by_severity)