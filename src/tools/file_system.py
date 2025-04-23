import logging
from langchain_core.tools import tool
from typing import Annotated,Union

from src.utils.file_system import list_dir,get_dir_metadata

from .decorators import log_io

logger = logging.getLogger(__name__)

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
