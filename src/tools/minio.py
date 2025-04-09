import logging
from langchain_core.tools import tool
from typing import Annotated

from src.utils import read,create,delete
from .decorators import log_io

logger = logging.getLogger(__name__)

@tool()
@log_io
def list_objects(prefix: Annotated[str, "Sub-folder name in minio bucket"]) -> list:
    """
    List objects in the bucket under the prefix.
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
def download_objects(prefix: Annotated[str, "Sub-folder name in minio bucket"]) -> bool:
    """
    Download all objects under a given prefix and preserve the folder structure locally.
    """
    try:
        return read.download_object(prefix)
    except Exception as e:
        # Catch any other exceptions
        error_message = f"Error executing command: {str(e)}"
        logger.error(error_message)
        return error_message

@tool()
@log_io
def upload_objects(file_path: Annotated[str, "Local file path to folder/filename"], prefix: Annotated[str, "Sub-folder name in minio bucket"]) -> bool:
    """
    Upload a single file or directory of files to the bucket at the given prefix.
    """
    try:
        return create.upload_object(file_path,prefix)
    except Exception as e:
        # Catch any other exceptions
        error_message = f"Error executing command: {str(e)}"
        logger.error(error_message)
        return error_message

@tool()
@log_io
def delete_objects(prefix: Annotated[str, "Sub-folder name in minio bucket"]) -> bool:
    """
    Delete all objects under a given prefix (simulating a folder).
    """
    try:
        return delete.delete_object(prefix)
    except Exception as e:
        # Catch any other exceptions
        error_message = f"Error executing command: {str(e)}"
        logger.error(error_message)
        return error_message


