import os
import logging
import warnings
warnings.filterwarnings("ignore")
from typing import List
logger = logging.getLogger(__name__)

from langchain_core.tools import tool
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology

from .decorators import log_io
from src.utils.minio import Create

@tool
@log_io
def grounded_annotator(classes: List[str]) -> bool:
    """
    Annotates images in the specified directory with bounding boxes and class labels.
    classes (List[str]): A list of class labels to use for annotation.
    """
    DATA_DIR = os.getenv("DATA_DIR")
    PROJECT_NAME = os.getenv("PROJECT_NAME")
    input_path = f"{DATA_DIR}/processed"
    os.makedirs(input_path,exist_ok=True)
    output_path = f"{DATA_DIR}/annotated"
    os.makedirs(output_path,exist_ok=True)

    classes = {k:v for k,v in zip(classes,classes)}
    try:
        import shutil

        base_model = GroundedSAM(ontology=CaptionOntology(classes))

        base_model.label(input_folder=input_path, extension=".jpg")
        default_out = f"{input_path}_labeled"

        # If user provided custom output_path, move/rename the default folder
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            # Move all labelled files into the custom directory
            for fname in os.listdir(default_out):
                shutil.move(os.path.join(default_out, fname), os.path.join(output_path, fname))
            shutil.rmtree(default_out, ignore_errors=True)
            logger.info(f"Annotated images moved to {output_path}")
        else:
            logger.info(f"Annotated images saved to {default_out}")
        create = Create()
        create.upload_object(output_path, f"{PROJECT_NAME}/annotated")
        logger.info(f"Annotated images uploaded to {PROJECT_NAME}/annotated in minio")
        return True
    except Exception as e:
        logger.error(f"Error in annotator: {e}")
        return False
    






    
    
    