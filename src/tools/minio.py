import logging
import os
from langchain_core.tools import tool
from typing import Annotated

from src.utils import read,create
from .decorators import log_io
from src.config import PATHS

logger = logging.getLogger(__name__)

@tool()
@log_io
def list_objects(prefix: Annotated[str, "prefix name in minio bucket"]) -> list:
    """
    List objects in the bucket under the prefix
    """
    try:
        return str(read.list_object(prefix))
    except Exception as e:
        # Catch any other exceptions
        error_message = f"Error executing command: {str(e)}"
        logger.error(error_message)
        return error_message

@tool()
@log_io
def download_objects(prefix: Annotated[str, "prefix name in minio bucket"]) -> bool:
    """
    Download all objects under a given prefix and preserve the folder structure locally.
    Prefix is the sub-directory name inside a minio bucket containing several files. 
    For example : Dataset/Sub-folder1/image1.png. Here, prefix = Dataset/Sub-folder1
    For example : DAWN/Fog/image1.jpg. Here, prefix = DAWN/Fog
    """
    logger.info(f"Actual prefix detected by llm : {prefix}")
    logger.info(f"Corrected prefix : {os.path.dirname(prefix)}")
    output_path = f"{PATHS['raw']}/{prefix}"
    logger.info(f"setting output path {output_path}")
    os.makedirs(output_path, exist_ok=True)
    try:
        if read.download_object(prefix,PATHS['raw']):
            return f"Downloaded objects to local file system of path : {PATHS['raw']}"
        else:
            return f"Failed to download objects to local file system of path"
    except Exception as e:
        # Catch any other exceptions
        error_message = f"Error executing command: {str(e)}"
        logger.error(error_message)
        return error_message

@tool()
@log_io
def upload_objects(input_path: Annotated[str, "Local directory path"], prefix: Annotated[str, "prefix name in minio bucket"]) -> bool:
    """
    Upload a single file or directory of files to the bucket at the given prefix.
    """
    try:
        return create.upload_object(input_path,prefix)
    except Exception as e:
        # Catch any other exceptions
        error_message = f"Error executing command: {str(e)}"
        logger.error(error_message)
        return error_message


