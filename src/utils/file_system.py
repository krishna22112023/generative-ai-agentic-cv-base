import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import datetime
import logging
import sys
import pyprojroot

root = pyprojroot.find_root(pyprojroot.has_dir("src"))
from src.config import PATHS

logger = logging.getLogger(__name__)


def create_dir(path: str) -> bool:
    """
    Creates a directory at the specified path.
    
    Args:
        path (str): The file system path of the directory to create.
        exist_ok (bool): If True, no exception will be raised if the directory exists.
                         Default is True.
    
    Returns:
        bool: True if the directory was created or already exists; False if an error occurred.
    
    Raises:
        OSError: If the directory cannot be created and exist_ok is False.
    """
    try:
        Path(root,path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {path}: {e}")
        return False

def list_dir() -> List[str]:
    """
    Lists the contents of the specified directory.
    
    Returns:
        List[str]: A list of file and directory names present in the specified directory.
    
    Raises:
        FileNotFoundError: If the directory does not exist.
        NotADirectoryError: If the specified path is not a directory.
    """
    path = Path(PATHS["data"])
    if not path.exists():
        logger.info(f"The directory {path} does not exist.")
        create_dir(path)
        return []
    if not path.is_dir():
        logger.error(f"The path {path} is not a directory.")
        return []
    
    return [item.name for item in path.iterdir()]

def delete_dir(path: str, force: bool = False) -> bool:
    """
    Deletes the directory at the specified path.
    
    Args:
        path (str): The file system path of the directory to delete.
        force (bool): If True, deletes the directory and all its contents recursively.
                      If False, only deletes the directory if it is empty.
                      Default is False.
    
    Returns:
        bool: True if the directory was successfully deleted, False otherwise.
    
    Raises:
        OSError: If the deletion fails.
    """
    try:
        if force:
            shutil.rmtree(Path(root,path))
        else:
            os.rmdir(Path(root,path))  # Only removes empty directories
        return True
    except Exception as e:
        print(f"Error deleting directory {path}: {e}")
        return False

def delete_file(path: str) -> bool:
    """
    Deletes the file at the specified path.
    
    Args:
        path (str): The file system path of the file to delete.
    
    Returns:
        bool: True if the file was successfully deleted; False otherwise.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If the deletion fails.
    """
    try:
        os.remove(Path(root,path))
        return True
    except Exception as e:
        print(f"Error deleting file {path}: {e}")
        return False

def get_dir_metadata(path: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves metadata about a directory.
    
    Args:
        path (str): The file system path of the directory.
    
    Returns:
        Dict[str, Any]: A dictionary containing metadata about the directory, including:
            - size (int): Total size of the directory contents in bytes.
            - num_files (int): Total number of files in the directory (recursively).
            - num_dirs (int): Total number of subdirectories (recursively).
            - creation_time (datetime): The creation time of the directory.
            - modification_time (datetime): The last modification time of the directory.
        Returns None if the directory does not exist or an error occurs.
    
    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    p = Path(root,'data',path)
    if not p.exists() or not p.is_dir():
        print(f"Directory {path} does not exist or is not a directory.")
        return None
    
    total_size = 0
    num_files = 0
    num_dirs = 0
    
    try:
        for item in p.rglob('*'):
            if item.is_file():
                num_files += 1
                total_size += item.stat().st_size
            elif item.is_dir():
                num_dirs += 1
                
        creation_time = datetime.datetime.fromtimestamp(p.stat().st_ctime)
        modification_time = datetime.datetime.fromtimestamp(p.stat().st_mtime)
        
        metadata = {
            "size": total_size,
            "num_files": num_files,
            "num_dirs": num_dirs,
            "creation_time": creation_time,
            "modification_time": modification_time,
        }
        
        return metadata
    except Exception as e:
        print(f"Error retrieving metadata for directory {path}: {e}")
        return None


