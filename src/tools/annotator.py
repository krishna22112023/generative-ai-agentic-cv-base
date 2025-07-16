import pyprojroot
import logging
import warnings
warnings.filterwarnings("ignore")
from typing import List
logger = logging.getLogger(__name__)

from langchain_core.tools import tool
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology

from .decorators import log_io

root = pyprojroot.find_root(pyprojroot.has_dir("src"))
DATA_DIR = os.getenv("DATA_DIR")

@tool
@log_io
def grounded_annotator(classes: List[str]) -> bool:
    """
    Annotates images in the specified directory with bounding boxes and class labels.
    classes (List[str]): A list of class labels to use for annotation.
    """
    input_path = f"{DATA_DIR}/processed_final"
    classes = {k:v for k,v in zip(classes,classes)}
    try:
        base_model = GroundedSAM(ontology=CaptionOntology(classes))

        base_model.label(input_folder=input_path, extension=".jpg")
        logger.info(f"Annotated images saved to {input_path}_labeled")
    
        return True
    except Exception as e:
        logger.error(f"Error in annotator: {e}")
        return False




    
    
    