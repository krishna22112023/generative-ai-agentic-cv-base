from __future__ import annotations
import os
import subprocess
import asyncio
import logging
from functools import partial
from typing import Dict
import re

import pyprojroot
from src.utils.minio import Read

_root = pyprojroot.find_root(pyprojroot.has_dir("src"))

logger = logging.getLogger(__name__)

_dataset_locks: Dict[str, asyncio.Lock] = {}

_TAG_SAFE_CHARS = r"[^0-9A-Za-z._-]"


def _slugify(name: str) -> str:
    """Return a git-safe slug (replace disallowed chars with '_')."""
    return re.sub(_TAG_SAFE_CHARS, "_", name.strip())


def _build_tag(project_name: str, version_id: str) -> str:
    """Return canonical git tag name for the dataset version.

    Git tag refs cannot contain spaces or many special chars. We slugify the
    *project_name* portion so that the tag is always valid while still
    reversibly encoding the original name (spaces -> '_', etc.)."""
    return f"{_slugify(project_name)}-v{version_id}"


def _run(cmd: list[str]) -> None:
    """Wrapper around subprocess.run with logging & error propagation."""
    logger.debug("Running command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

def dvc_add_commit_push(dataset_path: str, project_name: str, version_id: str) -> None:
    """Track dataset_path with DVC, push it to the default remote, and create a git tag.

    Args:
        dataset_path: str
            Local path to the dataset directory (e.g. 'data/my_project').
        project_name: str
            Name of the project (used in tag).
        version_id: str
            Version identifier (used in tag).
    """
    tag = _build_tag(project_name, version_id)

    # 1) dvc add <path>
    _run(["dvc", "add", dataset_path])

    # 2) git add the generated .dvc file and commit
    dvc_file = f"{dataset_path}.dvc"
    _run(["git", "add", dvc_file])
    _run(["git", "commit", "-m", f"Add {project_name} v{version_id}"])

    # 3) dvc push data to remote
    _run(["dvc", "push"])

    # 4) tag the commit and push tag
    _run(["git", "tag", "-f", tag])
    _run(["git", "push", "origin", "HEAD", "--tags"])


def dvc_pull(project_name: str, version_id: str) -> None:
    """Ensure *project_name*\@*version_id* exists locally via DVC.

    Strategy:
    1. Fetch the git tag (created by :func:`dvc_add_commit_push`).
    2. Temporarily check it out in detached-HEAD mode.
    3. Run ``dvc pull`` for the dataset path.
    4. Restore the previous branch/commit.
    """
    tag = _build_tag(project_name, version_id)
    dataset_rel_path = os.path.join("data", project_name)

    # Remember original HEAD so we can restore it afterwards.
    current_head = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
    ).strip()

    try:
        # Make sure the tag exists locally.
        _run(["git", "fetch", "origin", tag])

        # Checkout the tag (detached HEAD)
        _run(["git", "checkout", tag])

        # Pull the dataset files for that revision
        _run(["dvc", "pull", dataset_rel_path])

    finally:
        # Always return to the original HEAD (branch or commit).
        _run(["git", "checkout", current_head])


def download_from_minio(project_name: str, bucket_name: str) -> str:
    """Download dataset from MinIO and return the local path.
    
    Args:
        project_name: Name of the project (used as MinIO prefix)
        bucket_name: MinIO bucket name
        
    Returns:
        Local path where the dataset was downloaded
    """
    # Ensure top-level data directory exists (mirrors MinIO structure exactly)
    data_root = os.path.join(_root, "data")
    os.makedirs(data_root, exist_ok=True)

    # MinIO prefix (e.g. "My Project/") and local destination
    minio_prefix = f"{project_name}/"

    # Download using existing MinIO utility
    reader = Read()
    success = reader.download_object(minio_prefix, data_root)

    if not success:
        raise RuntimeError(
            f"Failed to download {project_name} from MinIO bucket {bucket_name}"
        )

    local_dataset_path = os.path.join(data_root, project_name)
    logger.info("Downloaded %s from MinIO to %s", project_name, local_dataset_path)
    return local_dataset_path


async def ensure_dataset_async(project_name: str, version_id: str | None, bucket_name: str | None = None) -> None:
    """Guarantee that *project_name* for *version_id* is present locally.

    Workflow:
    1. Try to pull existing version from DVC
    2. If that fails, download from MinIO and create new DVC version
    
    Args:
        project_name: Name of the project 
        version_id: Version identifier (if None, function returns immediately)
        bucket_name: MinIO bucket name (required if version doesn't exist in DVC)
    """
    if not version_id:
        return

    key = f"{project_name}:{version_id}"
    lock = _dataset_locks.setdefault(key, asyncio.Lock())

    async with lock:
        loop = asyncio.get_running_loop()
        
        # First, try to pull existing version
        try:
            await loop.run_in_executor(None, partial(dvc_pull, project_name, version_id))
            logger.info("Dataset %s@%s ready (pulled from DVC)", project_name, version_id)
            return
        except subprocess.CalledProcessError:
            logger.info("Version %s@%s not found in DVC, will download from MinIO", project_name, version_id)
        
        # If pull failed, download from MinIO and create new version
        if not bucket_name:
            raise ValueError(f"bucket_name is required to download new version {project_name}@{version_id}")
        
        try:
            # Download from MinIO (blocking operation)
            local_path = await loop.run_in_executor(
                None, partial(download_from_minio, project_name, bucket_name)
            )
            
            # Add to DVC and push (blocking operation)
            await loop.run_in_executor(
                None, partial(dvc_add_commit_push, local_path, project_name, version_id)
            )
            
            logger.info("Dataset %s@%s ready (downloaded from MinIO and versioned)", project_name, version_id)
            
        except Exception as exc:
            logger.error("Failed to download/version %s@%s: %s", project_name, version_id, exc)
            raise