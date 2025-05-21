"""Tool for Image restoration tasks using Restormer"""

from langchain_core.tools import tool

import os
import logging
import json
import sys
import subprocess
import shutil
import pyprojroot
from typing import List,Optional, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile

from .decorators import log_io
from src.config import PATHS,PREPROCESSOR_MODEL_MAP,MODEL_SCRIPT_CONFIG

root = pyprojroot.find_root(pyprojroot.has_dir("src"))
sys.path.append(str(root))

logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Define the log message format
    handlers=[
        logging.StreamHandler()  # Output logs to the console
    ]
)
logger = logging.getLogger(__name__)


@tool()
@log_io
def create_ir_pipeline(prefix:str,pipeline:Optional[str]) -> str:
    """_summary_

    Args:
        prefix (str): prefix name in minio bucket
        pipeline (Optional[str]): Optional custom pipeline from user to be applied to all images 

    Returns:
        str: summary of the pipeline to be executed
    """
    logger.info("Generating pipeline for image restoration tasks.")
    artefacts_path = f"{PATHS['artefacts']}/{prefix}"
    raw_path = f"{PATHS['raw']}/{prefix}"
    iqa_results_path = os.path.join(artefacts_path, "degredation_iqa_results.json")
    inferred_pipeline = {}
    if not pipeline:
        logger.info(f"No custom pipeline provided by user. Proceed to detect pipeline automatically.")
        if not os.path.exists(iqa_results_path):
            logger.error(f"Degredation IQA results file not found: {iqa_results_path}")
            logger.info(f"Skipping pipeline generation")
        else:
            #mapping the degredations to name of restormer models
            with open(iqa_results_path, 'r') as f:
                iqa_results = json.load(f)
            # Determine which restoration tools to run based on the IQA results
            for fname, degradations in iqa_results.items():
                model_pipeline = {model: [] for model in PREPROCESSOR_MODEL_MAP}
                for degradation in degradations:
                    if degradation.get("severity") not in ("medium", "high", "very high"):
                        continue
                    degradation_type = degradation.get("degradation")
                    for model, mapping in PREPROCESSOR_MODEL_MAP.items():
                        matched_models = mapping.get(degradation_type)
                        if matched_models:
                            model_pipeline[model].extend(matched_models)
                # Remove empty model lists
                inferred_pipeline[fname] = {k: v for k, v in model_pipeline.items() if v}
    else:
        logger.info(f"Custom pipeline provided: {pipeline}")
        for fname in os.listdir(raw_path):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in ['.jpg', '.jpeg', '.png']:
                continue
            model_pipeline = {model: [] for model in PREPROCESSOR_MODEL_MAP}
            for degradation_type in pipeline:
                for model, mapping in PREPROCESSOR_MODEL_MAP.items():
                    matched_models = mapping.get(degradation_type)
                    if matched_models:
                        model_pipeline[model].extend(matched_models)
            inferred_pipeline[fname] = {k: v for k, v in model_pipeline.items() if v}

    # Save the pipeline to a JSON file
    output_pipeline_path = os.path.join(artefacts_path, "preprocessing_pipeline.json")
    with open(output_pipeline_path, 'w', encoding='utf-8') as outfile:
        json.dump(inferred_pipeline, outfile, indent=4)
    logger.info(f"Pipeline generated and saved to {output_pipeline_path}")

    return json.dumps(inferred_pipeline)

def process_single_image(prefix: str, filename: str) -> None:
    """
    Process one image through every model/task chain:
      a) Copy raw image into its own temp folder.
      b) For each model, chain all its tasks in that temp folder.
      c) Copy final outputs back to processed/<prefix>/<image>/models/<model>/<filename>.

    args:
      prefix   – The dataset subfolder (e.g. "Test").
      filename – e.g. "foggy-001.jpg"
    """
    # Build all the key paths
    raw_folder      = os.path.join(PATHS["raw"], prefix)
    artefacts_folder= os.path.join(PATHS["artefacts"], prefix)
    proc_root       = os.path.join(PATHS["processed"], prefix)
    proc_final      = os.path.join(PATHS["processed_final"], prefix)

    # Read this image’s entry in the pipeline JSON
    pipeline_path = os.path.join(artefacts_folder, "preprocessing_pipeline.json")
    with open(pipeline_path, "r", encoding="utf-8") as f:
        pipeline = json.load(f)
    if filename not in pipeline:
        logger.warning(f"{filename} not in pipeline spec, skipping")
        return
    model_tasks = pipeline[filename]

    # Create a temp workspace and copy raw image in
    with tempfile.TemporaryDirectory() as workdir:
        src_img = os.path.join(raw_folder, filename)
        if not os.path.isfile(src_img):
            logger.error(f"Raw file not found: {src_img}")
            return

        # copy2 preserves metadata
        work_img = os.path.join(workdir, filename)
        shutil.copy2(src_img, work_img)
        current_input_dir = workdir

        if not model_tasks:
            logger.warning(f"No tasks for {filename}; skipping and saving to final preprocessed folder")
            src = os.path.join(raw_folder, filename)
            dest = os.path.join(proc_final, filename)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(src, dest)

        # 3️⃣ For each model, run its full chain in the same temp dir
        for model_name, tasks in model_tasks.items():
            cfg = MODEL_SCRIPT_CONFIG.get(model_name)
            if not cfg:
                logger.warning(f"No config for model {model_name}; skipping")
                continue

            logger.info(f"[{filename}] → model {model_name}: tasks {tasks}")
            for task in tasks:
                # build the CLI depending on model
                if model_name == "restormer":
                    cmd = [
                        "conda", "run", "-n", cfg["env"], cfg["python"],
                        os.path.join(root, "modules", cfg["script"]),
                        "--task", task,
                        "--input_dir", current_input_dir,
                        "--result_dir", workdir
                    ]
                elif model_name == "swinir":
                    if "color_dn" in task:
                        noise = task.split("_")[-1]
                        task = "color_dn"
                        cmd = [
                            "conda", "run",
                            "-n", cfg["env"],
                            cfg["python"],
                            f"{root}/modules/{cfg['script']}",
                            "--task", task,
                            "--noise", noise,
                            "--folder_lq", current_input_dir,
                            "--save_dir", workdir,
                        ]
                    elif "color_jpeg" in task:
                        jpeg = task.split("_")[-1]
                        task = "color_jpeg"
                        cmd = [
                            "conda", "run",
                            "-n", cfg["env"],
                            cfg["python"],
                            f"{root}/modules/{cfg['script']}",
                            "--task", task,
                            "--jpeg", jpeg,
                            "--folder_lq", current_input_dir,
                            "--save_dir", workdir,
                        ]
                    elif "real_sr" in task:
                        cmd = [
                            "conda", "run",
                            "-n", cfg["env"],
                            cfg["python"],
                            f"{root}/modules/{cfg['script']}",
                            "--task", task,
                            "--folder_lq", current_input_dir,
                            "--save_dir", workdir,
                            "--scale", "4",
                        ]
                elif model_name == "xrestormer":
                    cmd = [
                        "conda", "run", "-n", cfg["env"], cfg["python"],
                        os.path.join(root, "modules", cfg["script"]),
                        "-opt", os.path.join(root, "src", "config", "xrestormer", f"{task}.yml"),
                        "--dataroot_lq", current_input_dir,
                        "--results_dir", workdir
                    ]
                else:
                    logger.error(f"Unknown model {model_name} or no preprocessing specified.")
                    break

                # run the task
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    logger.error(f"[{filename}] {model_name}/{task} failed: {e.stderr.strip()}")
                    break

                # after each task, workdir now contains the new image as filename
                current_input_dir = workdir

            # Copy final output into your processed tree
            dest_dir = os.path.join(proc_root,
                                    os.path.splitext(filename)[0],
                                    "models",
                                    model_name)
            os.makedirs(dest_dir, exist_ok=True)
            final_src = os.path.join(workdir, filename)
            if os.path.isfile(final_src):
                shutil.copy2(final_src, os.path.join(dest_dir, filename))
                logger.info(f"[{filename}] → saved {model_name} output to {dest_dir}")
            else:
                logger.warning(f"[{filename}] no final output for {model_name}")

@tool()
@log_io
def run_ir_pipeline(prefix: str) -> bool:
    """
    Dispatch all images in parallel through `process_single_image`.
    Returns False only if pipeline spec is missing.
    """
    max_workers = 2
    artefacts_folder = os.path.join(PATHS["artefacts"], prefix)
    raw_folder       = os.path.join(PATHS["raw"], prefix)

    pipeline_path = os.path.join(artefacts_folder, "preprocessing_pipeline.json")
    if not os.path.exists(pipeline_path):
        logger.error(f"Missing pipeline JSON at {pipeline_path}; aborting.")
        return False

    # list only supported image files
    all_images = [f for f in os.listdir(raw_folder)
                  if f.lower().endswith(('.jpg','.jpeg','.png'))]

    # parallel map: per‐image isolation
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(process_single_image, prefix, img): img
                   for img in all_images}
        for fut in as_completed(futures):
            img = futures[fut]
            try:
                fut.result()
                logger.info("All images processed.")
                return True
            except Exception as e:
                logger.error(f"Unhandled error on {img}: {e}")
                return False

    

if __name__ == "__main__":
    prefix = "Test"
    run_ir_pipeline(prefix)