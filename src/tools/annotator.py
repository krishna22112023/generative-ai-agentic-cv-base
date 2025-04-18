import os
from PIL import Image
import pyprojroot
import sys
import glob
import json
import logging
from typing import List
logger = logging.getLogger(__name__)


from langchain_core.tools import tool
from google import genai
from src.config.tools import ANNOTATION_API_KEY,ANNOTATION_MODEL

from .decorators import log_io
from src.utils.annotations import convert_annotations,get_processed_files

root = pyprojroot.find_root(pyprojroot.has_dir("src"))
sys.path.append(str(root))

@tool
@log_io
def gemini_annotator(prefix:str, classes: List[str], format: str) -> bool:
    """
    Annotates images in the specified directory with bounding boxes and class labels.
    prefix (str): The directory prefix where images are located.
    classes (List[str]): A list of class labels to use for annotation.
    format (str): The output format for the annotations. Default format is "yolo".

    """
    ## TO DO : Support more annotation formats
    if format.lower() not in ["yolo"]:
        format = "yolo"

    #get files 
    try:
        files = get_processed_files(f"{root}/data/processed/{prefix}")
    except Exception as e:
        msg = f"Error while retrieving processed files: {e}"
        logger.warning(msg)
        logger.info("Trying a different root path")
        files = glob.glob(f"{root}/data/raw/{prefix}/*.jpg")

    logger.info({f"Drawing bounding boxes for {len(files)} images"})

    try: 
        client = genai.Client(api_key=ANNOTATION_API_KEY)

        with open(f"{root}/src/prompts/gemini_annotator.txt", 'r', encoding='utf-8') as f:
            prompt = f.read()
        prompt = prompt.replace("<<CLASSES>>", ", ".join(classes))

        out_dir = f"{root}/data/processed/{prefix}/annotations"
        os.makedirs(out_dir, exist_ok=True)

        class_map = {name.lower(): idx for idx, name in enumerate(classes)}
        with open(out_dir + "/classes.txt", "w") as f:
            for class_name in classes:
                f.write(f"{class_name.lower()}\n")

        for file in files:
            file_ref = client.files.upload(file=file)
            response = client.models.generate_content(
                                    model=ANNOTATION_MODEL,
                                    contents=[file_ref, prompt])
            image_size = Image.open(file).size
            full_response = response.text
            if full_response.startswith("```json"):
                full_response = full_response.removeprefix("```json")
            if full_response.endswith("```"):
                full_response = full_response.removesuffix("```")
            annotations = convert_annotations(annotations=json.loads(full_response), class_map=class_map, image_size=image_size, format=format)
            if format == "yolo":
                with open(out_dir + "/" + os.path.basename(file).split('.')[0] + ".txt", "w") as f:
                    for ann in annotations:
                        f.write(f"{ann}\n")
        return True
    except Exception as e:
        msg = f"Error annotating images: {e}"
        logger.error(msg)
        return False                    

if __name__ == "__main__":
    gemini_annotator("DAWN/Rain", ["Car"], "yolo")