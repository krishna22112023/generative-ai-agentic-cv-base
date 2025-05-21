import os
import logging
from langchain_core.tools import tool
from typing import Annotated,Union

from src.utils.file_system import list_dir,get_dir_metadata,get_latest_ext_folder,process_latest_folder, organize_extracted_files
from src.config import PATHS

from .decorators import log_io

logger = logging.getLogger(__name__)

@tool
@log_io
def extract_dir_local(sample_size:int):
    """Extracts or Unzips data from raw datasets downloaded from external sources and flattens them into a unified directory if required. Eg. huggingface
    Supported file formats include .zip, .tar, .parquet

    Args:
        sample_size (int): The number of files to sample from the flattened directory.

    Returns:
        str: The data directory path.
    """
    input_path = PATHS["downloads"]
    output_path = PATHS["raw"]
    
    #Select folder with latest time stamp
    latest_ext_folder = get_latest_ext_folder(input_path)

    #Extract/Unzip folder from zip, tar or parquet
    extracted_folder = process_latest_folder(latest_ext_folder)

    #Flatten/Organize folder structure if required
    prefix = os.path.basename(latest_ext_folder)
    summary = organize_extracted_files(extracted_folder,output_path,prefix,sample_size)

    return summary, prefix

@tool
@log_io
def list_dir_local() -> Union[list,bool]:
    """list down folders within the data directory of local file system

    Returns:
        Union[list,bool]: List of directories present or False if error occured
    """
    dir_list = list_dir()
    if dir_list:
        return dir_list
    else:
        return False

@tool
@log_io
def get_dir_metadata_local(path: Annotated[str, "path to data directory in local file system"]) -> Union[dict,bool]:
    """_summary_

    Args:
        path (Annotated[str, &quot;Sub):  relative path to data directory in local file system

    Returns:
        Union[dict,bool]: A dictionary containing metadata about the directory, including:
            - size (int): Total size of the directory contents in bytes.
            - num_files (int): Total number of files in the directory (recursively).
            - num_dirs (int): Total number of subdirectories (recursively).
            - creation_time (datetime): The creation time of the directory.
            - modification_time (datetime): The last modification time of the directory.
        Returns False if the directory does not exist or an error occurs.
    """
    metadata = get_dir_metadata(path)
    if metadata:
        return metadata
    else:
        return False
