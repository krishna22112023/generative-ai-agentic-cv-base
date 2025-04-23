import warnings
warnings.filterwarnings("ignore")
import pyprojroot
from pydantic import BaseModel, Field
from typing import List, Tuple, Dict
import os
import base64
from tqdm import tqdm
import json
import logging
import cv2
from PIL import Image
import numpy as np
import torch
from transformers import AutoModelForCausalLM

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder, SystemMessagePromptTemplate
)
from src.agents.llm import get_llm_by_type
from src.utils.preprocess import resize_image
root = pyprojroot.find_root(pyprojroot.has_dir("src"))

logger = logging.getLogger(__name__)

class Degradation(BaseModel):
    degradation: str = Field(..., description="One of the seven degradation types.")
    thought: str = Field(..., description="The assessment thought for the degradation.")
    severity: str = Field(..., description="Severity rating (very low, low, medium, high, very high).")

class IQAResponse(BaseModel):
    items: List[Degradation]

def get_perception_model():

    llm = get_llm_by_type("vision")

    # Inject the output parser’s instructions into your system prompt.
    fp_prompt = f"{root}/src/prompts/vlm_nr_iqa.md"
    with open(fp_prompt, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    system_template = SystemMessagePromptTemplate.from_template(prompt_template)
    # system_template = hub.pull(agent_config.LANGSMITH_PROMPT_TEMPLATE_NAME)

    messages = MessagesPlaceholder(variable_name='messages')
    metrics = MessagesPlaceholder(variable_name="metrics")

    prompt = ChatPromptTemplate.from_messages([system_template, metrics, messages])
    chain = prompt | llm.with_structured_output(schema=IQAResponse)

    return chain

def vlm_nr_iqa(input_path:str,artefacts_path:str,nr_iqa_scores:dict=None, extensions: List[str] = None) -> str:
    """Assess the quality of images in a directory using a pre-trained model based on 7 degredations that include 
    noise, motion blur, defocus blur, haze, rain, dark, and jpeg compression artifact and classify them into 5 severity levels 
    namely "very low", "low", "medium", "high", and "very high".
    
    Args:
        prefix (str): relative path within the data/raw or data/processed folder where the images are stored and need to be assessed. Eg : data/raw/prefix where prefix is the folder path in minio bucket

    Returns:
        str: json string containing the IQA results
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png']

    chain = get_perception_model()

    results = {}
    for fname in os.listdir(input_path):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in extensions:
            continue

        img_path = os.path.join(input_path, fname)
        image_data = resize_image(img_path)
        encoded_data = base64.b64encode(image_data).decode('utf-8')
        message = [{"role": "user", "content": f"data:image/jpeg;base64,{encoded_data}"}]
        response = chain.invoke({"messages": message, "metrics": nr_iqa_scores})
        try:
            parsed = response.model_dump()["items"]
            # Filter out degradations with severity "high" or "very high"
            #filtered = [item for item in parsed if item.get("severity") in ("high", "very high")]
            results[fname] = parsed
        except json.JSONDecodeError as e:
            results[fname] = {"error": "JSON decode error", "detail": str(e)}
        
    # Save raw results to iqa_results.json
    with open(os.path.join(artefacts_path, "openai_iqa_results.json"), 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, indent=4)
    
    # Columns are degradation types; rows are severity levels.
    degradations = ["noise", "motion blur", "defocus blur", "haze", "rain", "dark", "jpeg compression artifact"]
    severities = ["very low", "low", "medium","high", "very high"]
    aggregated = {sev: {deg: 0 for deg in degradations} for sev in severities}
    
    for items in results.values():
        if isinstance(items, list):
            for item in items:
                sev = item.get("severity")
                deg = item.get("degradation")
                if sev in severities and deg in degradations:
                    aggregated[sev][deg] += 1
    with open(os.path.join(artefacts_path, "openai_iqa_results_aggregated.json"), 'w', encoding='utf-8') as outfile:
        json.dump(aggregated, outfile, indent=4)

    return json.dumps(aggregated, indent=4)

def nr_iqa(
    input_path: str,
    artefacts_path: str,
    extensions: List[str] = None,
    metrics: List[str] = ['brisque','qalign']
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Compute no-reference image quality assessment metrics for all matching images in a directory.

    Args:
        input_path (str): Path to the folder containing the distorted images.
        metrics (List[str]): List of metrics to compute. Supported: 'brisque', 'qalign'.
        artefacts_path (str): (Unused) Path to save any artefacts or logs.
        ref_path (str): Path to the folder containing reference images with matching filenames.
        extensions (List[str], optional): List of file extensions to consider. Defaults to ['.jpg', '.jpeg', '.png'].

    Returns:
        scores (Dict[str, Dict[str, float]]): Nested dict mapping filename -> metric -> score.
        mean_scores (Dict[str, float]): Average score per metric across all processed images.

    Raises:
        ValueError: If no valid images are processed or if any requested metric is unsupported.
    """
    # collect metadata
    stats = get_metadata(input_path,artefacts_path)
    logger.info("dataset metadata collected")

    # Supported metrics
    supported_metrics = ['brisque','qalign']

    # Normalize and validate requested metrics
    metrics = [m.lower() for m in metrics]
    for m in metrics:
        if m not in supported_metrics:
            raise ValueError(f"Unsupported metric '{m}'. Supported: {supported_metrics}")

    # Default extensions
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png']

    
    brisque_model = cv2.quality.QualityBRISQUE_create("src/config/brisque_model_live.yml","src/config/brisque_range_live.yml")
    qalign_model = AutoModelForCausalLM.from_pretrained("q-future/one-align", trust_remote_code=True, attn_implementation="eager", 
                                        torch_dtype=torch.float16, device_map="auto")

    # Prepare output structures
    scores: Dict[str, Dict[str, float]] = {}

    # Iterate over images in input directory
    for fname in tqdm(os.listdir(input_path),desc="No Reference Image Quality Assessment Running"):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in extensions:
            continue

        img_path = os.path.join(input_path, fname)
        brisque_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        qalign_img = Image.open(img_path)
        
        # Initialize nested dict for this file
        scores[fname] = {}

        # Compute each requested metric
        for metric in metrics:
            if metric == 'brisque':
                try:
                    out = brisque_model.compute(brisque_img)
                    score_value = float(out[0]) if isinstance(out, (tuple, list)) else float(out)
                except Exception as e:
                    logger.info(f"An error occured inferencing BRISQUE. Skipping image. {e}")
                    score_value = None
            elif metric == 'qalign':
                try:
                    out = qalign_model.score([qalign_img], task_="quality", input_="image")
                    score_value = float(out.detach().cpu().numpy()[0])
                except Exception as e:
                    logger.info(f"An error occured inferencing Q-Align. Skipping image. {e}")
                    score_value = None
            else:
                score_value = None
            # Handle different return types
            scores[fname][metric] = score_value

    # Ensure we processed at least one image
    if not scores:  
        raise ValueError(f"No valid images processed in '{input_path}'.")

    # Compute mean scores per metric
    mean_scores: Dict[str, float] = {}
    num_images = len(scores)
    for metric in metrics:
        total = sum(img_scores[metric] for img_scores in scores.values())
        mean_scores[metric] = total / num_images

    with open(os.path.join(artefacts_path, "nr_iqa_results.json"), 'w', encoding='utf-8') as outfile:
        json.dump(scores, outfile, indent=4)
    return scores, mean_scores


def fr_iqa(
    input_path: str,
    artefacts_path: str,
    ref_path: str,
    extensions: List[str] = None,
    metrics: List[str]=['psnr', 'ssim', 'mse', 'gmsd']
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Compute full-reference image quality assessment metrics for all matching images in a directory.

    Args:
        input_path (str): Path to the folder containing the distorted images.
        metrics (List[str]): List of metrics to compute. Supported: 'psnr', 'ssim', 'mse', 'gmsd'.
        artefacts_path (str): (Unused) Path to save any artefacts or logs.
        ref_path (str): Path to the folder containing reference images with matching filenames.
        extensions (List[str], optional): List of file extensions to consider. Defaults to ['.jpg', '.jpeg', '.png'].

    Returns:
        scores (Dict[str, Dict[str, float]]): Nested dict mapping filename -> metric -> score.
        mean_scores (Dict[str, float]): Average score per metric across all processed images.

    Raises:
        ValueError: If no valid images are processed or if any requested metric is unsupported.
    """
    # Supported metrics
    supported_metrics = ['psnr', 'ssim', 'mse', 'gmsd']

    # Normalize and validate requested metrics
    metrics = [m.lower() for m in metrics]
    for m in metrics:
        if m not in supported_metrics:
            raise ValueError(f"Unsupported metric '{m}'. Supported: {supported_metrics}")

    # Default extensions
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png']

    # Prepare output structures
    scores: Dict[str, Dict[str, float]] = {}

    # Iterate over images in input directory
    for fname in tqdm(os.listdir(input_path)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in extensions:
            continue

        img_path = os.path.join(input_path, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.info(f"Unable to read image {img_path}, skipping.")
            continue

        # Construct reference image path
        ref_img_path = os.path.join(ref_path, fname)
        ref = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)
        if ref is None:
            logger.info(f"Unable to read reference image {ref_img_path}, skipping.")
            continue

        # Initialize nested dict for this file
        scores[fname] = {}

        # Compute each requested metric
        for metric in metrics:
            # Create the appropriate analyzer per metric
            if metric == 'psnr':
                analyzer = cv2.quality.QualityPSNR_create(ref)
            elif metric == 'mse':
                analyzer = cv2.quality.QualityMSE_create(ref)
            elif metric == 'ssim':
                analyzer = cv2.quality.QualitySSIM_create(ref)
            elif metric == 'gmsd':
                analyzer = cv2.quality.QualityGMSD_create(ref)
            else:
                # This should never happen due to earlier validation
                continue

            # Compute and store the score
            out = analyzer.compute(img)
            # Handle different return types
            score_value = float(out[0]) if isinstance(out, (tuple, list)) else float(out)
            scores[fname][metric] = score_value

    # Ensure we processed at least one image
    if not scores:
        raise ValueError(f"No valid images processed in '{input_path}'.")

    # Compute mean scores per metric
    mean_scores: Dict[str, float] = {}
    num_images = len(scores)
    for metric in metrics:
        total = sum(img_scores[metric] for img_scores in scores.values())
        mean_scores[metric] = total / num_images

    with open(os.path.join(artefacts_path, "fr_iqa_results.json"), 'w', encoding='utf-8') as outfile:
        json.dump(scores, outfile, indent=4)
    return scores, mean_scores

def get_metadata(input_path, output_path):
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

    if output_path:
        with open(output_path+'/metadata.json', 'w') as f:
            json.dump(stats, f, indent=2)

    return stats

