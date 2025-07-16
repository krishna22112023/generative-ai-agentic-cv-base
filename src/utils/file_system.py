import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import datetime
import logging
import glob
import pyprojroot
import re
import pandas as pd
import requests
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor

root = pyprojroot.find_root(pyprojroot.has_dir("src"))
DATA_DIR = os.getenv("DATA_DIR")

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
    path = DATA_DIR
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
        logger.info(f"Error deleting directory {path}: {e}")
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
        logger.info(f"Error deleting file {path}: {e}")
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
        logger.info(f"Directory {path} does not exist or is not a directory.")
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
        logger.info(f"Error retrieving metadata for directory {path}: {e}")
        return None

def get_latest_ext_folder(base_dir):
    # Get all folders matching the ext_* pattern
    ext_folders = glob.glob(os.path.join(base_dir, "ext_*"))
    
    # Extract timestamps and associate with paths
    folder_timestamps = []
    for folder in ext_folders:
        # Extract the timestamp part from the folder name
        folder_name = os.path.basename(folder)
        match = re.match(r"ext_(\d{8}_\d{6})", folder_name)
        if match:
            timestamp_str = match.group(1)
            try:
                # Convert timestamp string to datetime object
                timestamp = datetime.datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                folder_timestamps.append((timestamp, folder))
            except ValueError:
                # Skip folders with invalid timestamp format
                continue
    
    # Find the folder with the latest timestamp
    if folder_timestamps:
        latest_folder = max(folder_timestamps, key=lambda x: x[0])[1]
        return latest_folder
    else:
        return None

def process_latest_folder(folder_path):
    """
    Process files in the given folder based on their extensions.
    Extract archives and download images from parquet files.
    """
    # Create a subfolder for extraction
    extraction_subfolder = os.path.join(folder_path, "extracted_data")
    os.makedirs(extraction_subfolder, exist_ok=True)
    
    # Process all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Skip if it's a directory or not a file
        if not os.path.isfile(file_path):
            continue
            
        # Handle .tar.gz files
        if filename.endswith('.tar.gz'):
            logger.info(f"Extracting tar.gz archive: {filename}")
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(path=extraction_subfolder)
                
        # Handle .zip files
        elif filename.endswith('.zip'):
            logger.info(f"Extracting zip archive: {filename}")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extraction_subfolder)
                
        # Handle .parquet files
        elif filename.endswith('.parquet'):
            logger.info(f"Processing parquet file: {filename}")
            images_folder = os.path.join(extraction_subfolder, f"{os.path.splitext(filename)[0]}_images")
            os.makedirs(images_folder, exist_ok=True)
            
            # Extract images from parquet file
            extract_images_from_parquet(file_path, images_folder)
    
    return extraction_subfolder

def extract_images_from_parquet(parquet_file, output_folder):
    """
    Extract image URLs from a parquet file and download the images.
    """
    # Read the parquet file
    df = pd.read_parquet(parquet_file)
    
    # Look for columns that might contain image URLs
    image_columns = []
    
    # Common image column names to check first
    common_image_columns = ['image', 'image_url', 'img', 'img_url', 'image_path', 'url', 'image_link', 'picture']
    
    # First check common names
    for col in common_image_columns:
        if col in df.columns:
            image_columns.append(col)
    
    # If no common names found, use heuristics to detect columns containing URLs
    if not image_columns:
        for column in df.columns:
            # Check first few non-null values in each column
            sample_values = df[column].dropna().head(5).astype(str).tolist()
            
            # Check if values look like URLs and have image extensions
            url_pattern = re.compile(r'https?://\S+\.(jpg|jpeg|png|gif|bmp|webp)', re.IGNORECASE)
            
            if any(url_pattern.search(val) for val in sample_values):
                image_columns.append(column)
    
    logger.info(f"Found potential image columns: {image_columns}")
    
    # Download images from identified columns
    if image_columns:
        for column in image_columns:
            download_images_from_column(df, column, output_folder)
    else:
        logger.info("No image columns identified in the parquet file.")

def download_images_from_column(df, column, output_folder):
    """
    Download images from URLs in the specified column using multithreading.
    """
    urls = df[column].dropna().unique().tolist()
    logger.info(f"Found {len(urls)} unique image URLs in column '{column}'")
    
    def download_image(url):
        try:
            if not isinstance(url, str) or not url.startswith(('http://', 'https://')):
                return
                
            # Extract filename from URL or generate one
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            
            # If no filename, use the URL hash
            if not filename or '.' not in filename:
                filename = f"image_{hash(url) % 10000}.jpg"
            
            output_path = os.path.join(output_folder, filename)
            
            # Skip if file already exists
            if os.path.exists(output_path):
                return
                
            # Download the image
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
        except Exception as e:
            logger.info(f"Error downloading {url}: {str(e)}")
    
    # Use ThreadPoolExecutor to download images in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(download_image, urls)

def organize_extracted_files(
    extraction_path: str, 
    output_path: str, 
    prefix: str,
    sample_size: Optional[int] = None
):
    """
    Organize extracted files based on the specified options.
    
    Args:
        extraction_path: Path where files were extracted.
        output_path: Path where files should be organized.
        sample_size: If provided and flatten is True, randomly select this many files.
    
    Returns:
        A dictionary with statistics about the organized files.
    """
    stats = {"total_files": 0, "image_files": 0, "organized_files": 0}
    
    # Define image extensions to look for
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
    
    # Check if dataset has train/val/test structure
    dataset_structure = []
    for item in os.listdir(extraction_path):
        item_path = os.path.join(extraction_path, item)
        if os.path.isdir(item_path) and item.lower() in ['train', 'test', 'val', 'validation', 'dev']:
            dataset_structure.append(item)
    
    has_splits = len(dataset_structure) > 0
    logger.info(f"Dataset splits detected: {dataset_structure}" if has_splits else "No standard dataset splits detected")
    
    # Get all image files from extraction path recursively
    all_image_files = []
    
    for root, _, files in os.walk(extraction_path):
        for file in files:
            stats["total_files"] += 1
            _, ext = os.path.splitext(file.lower())
            if ext in image_extensions:
                stats["image_files"] += 1
                all_image_files.append(os.path.join(root, file))
    
    logger.info(f"Found {stats['image_files']} image files out of {stats['total_files']} total files")
    
    # Sample if needed
    if sample_size and len(all_image_files) > sample_size:
        selected_files = all_image_files[:sample_size]
        logger.info(f"Selected {sample_size} images from {len(all_image_files)} total images")
    else:
        selected_files = all_image_files
        if sample_size:
            logger.info(f"Sample size {sample_size} is larger than available images ({len(all_image_files)}). Using all images.")
    
    os.makedirs(os.path.join(output_path,prefix), exist_ok=True)
    # Copy files to output directory with flattened structure
    for i, src_path in enumerate(selected_files):
        # Create a unique filename to avoid collisions when flattening
        filename = os.path.basename(src_path)
        base_name, ext = os.path.splitext(filename)
        
        # Add a unique identifier to prevent name collisions
        unique_path = os.path.join(output_path,prefix,f"{base_name}_{i}{ext}")
        
        shutil.copy2(src_path, unique_path)
        stats["organized_files"] += 1
        
    logger.info(f"Flattened directory structure. Copied {stats['organized_files']} files to {output_path}")
    
    return stats, prefix