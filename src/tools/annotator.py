import pyprojroot
import logging
import warnings
warnings.filterwarnings("ignore")
from typing import Dict
logger = logging.getLogger(__name__)

from langchain_core.tools import tool
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology

from .decorators import log_io
from src.config import PATHS

root = pyprojroot.find_root(pyprojroot.has_dir("src"))

@tool
@log_io
def annotator(prefix:str, classes: Dict[str]) -> bool:
    """
    Annotates images in the specified directory with bounding boxes and class labels.
    prefix (str): The directory prefix where images are located.
    classes (List[str]): A list of class labels to use for annotation.
    """
    input_path = f"{PATHS['processed']}/{prefix}"

    try:
        base_model = GroundedSAM(ontology=CaptionOntology(classes))

        #results = base_model.predict("/Users/krishnaiyer/generative-ai-agentic-cv-base/data/processed/DAWN/Fog/foggy-003.jpg")
        # can send back as human in the loop element to confirm with user if detection is ok

        base_model.label(input_folder=input_path, extension=".jpg")
        logger.info(f"Annotated images saved to {input_path}_labeled")
    
        return True
    except Exception as e:
        logger.error(f"Error in annotator: {e}")
        return False




    
    
    