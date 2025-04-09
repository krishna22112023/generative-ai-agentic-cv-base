import os
import boto3
import logging
from typing import Dict, Any, List
from botocore.exceptions import ClientError

from src.config import MINIO_ENDPOINT_URL,MINIO_ACCESS_KEY,MINIO_SECRET_KEY,BUCKET_NAME

import pyprojroot
root = pyprojroot.find_root(pyprojroot.has_dir("config"))

logger = logging.getLogger(__name__)

LOCAL_DIR = "data/raw"

class Read:

    def __init__(self):
        self.client = boto3.client('s3',
                                endpoint_url=MINIO_ENDPOINT_URL,
                                aws_access_key_id=MINIO_ACCESS_KEY,
                                aws_secret_access_key=MINIO_SECRET_KEY)

    def list_object(self, prefix: str) -> List[Dict[str, Any]]:
        """
        List objects in the bucket. Use the prefix to emulate folder listing.
        """
        try:
            response = self.client.list_objects_v2(
                Bucket=BUCKET_NAME, 
                Prefix=prefix, 
                MaxKeys=1000
            )
            objects = response.get('Contents', [])
            logger.info(f"Found {len(objects)} objects with prefix '{prefix}' in bucket {BUCKET_NAME}")
            return objects
        except ClientError as e:
            logger.error(f"Error listing objects with prefix '{prefix}' in bucket {BUCKET_NAME}: {e}")
            return []

    def download_object(self, prefix: str) -> bool:
        """
        Download objects under the given prefix.
        If the prefix exactly matches a single file key, download that file.
        If the prefix represents a directory (matches multiple objects), download all objects under it,
        preserving the folder structure locally.
        """
        objects = self.list_object(prefix)
        if not objects:
            logger.error(f"No objects found with prefix '{prefix}'.")
            return False

        # If the prefix exactly matches a single file key, download that file.
        if len(objects) == 1 and objects[0].get('Key') == prefix:
            key = objects[0].get('Key')
            local_file_path = os.path.join(root, LOCAL_DIR)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            try:
                self.client.download_file(Bucket=BUCKET_NAME, Key=key, Filename=local_file_path)
                logger.info(f"Downloaded {key} to {local_file_path}")
                return True
            except ClientError as e:
                logger.error(f"Error downloading {key}: {e}")
                return False
        else:
            # Treat the prefix as a directory and download all objects.
            overall_success = True
            for obj in objects:
                key = obj.get('Key')
                local_file_path = os.path.join(root, LOCAL_DIR)
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                try:
                    self.client.download_file(Bucket=BUCKET_NAME, Key=key, Filename=local_file_path)
                    logger.info(f"Downloaded {key} to {local_file_path}")
                except ClientError as e:
                    logger.error(f"Error downloading {key}: {e}")
                    overall_success = False
            return overall_success

class Create:

    def __init__(self):
        self.client = boto3.client('s3',
                                endpoint_url=MINIO_ENDPOINT_URL,
                                aws_access_key_id=MINIO_ACCESS_KEY,
                                aws_secret_access_key=MINIO_SECRET_KEY)

    def upload_object(self, file_path: str, key: str) -> bool:
        """
        Upload a single file to the bucket at the given key.
        """
        # If file_path is a directory, iterate through its files recursively.
        if os.path.isdir(file_path):
            all_success = True
            for root_dir, dirs, files in os.walk(file_path):
                for file in files:
                    full_file_path = os.path.join(root_dir, file)
                    # Obtain the relative path of the file with respect to file_path
                    rel_path = os.path.relpath(full_file_path, start=file_path)
                    # Construct the destination key using the provided key as the base
                    dest_key = os.path.join(key, rel_path).replace("\\", "/")
                    try:
                        self.client.upload_file(Filename=full_file_path, Bucket=BUCKET_NAME, Key=dest_key)
                        logger.info(f"Uploaded {full_file_path} to bucket {BUCKET_NAME} as {dest_key}")
                    except ClientError as e:
                        logger.info(f"Error uploading {full_file_path} to bucket {BUCKET_NAME}: {e}")
                        all_success = False
            return all_success

        # If file_path is a file, upload it directly.
        elif os.path.isfile(file_path):
            try:
                self.client.upload_file(Filename=file_path, Bucket=BUCKET_NAME, Key=key)
                logger.info(f"Uploaded {file_path} to bucket {BUCKET_NAME} as {key}")
                return True
            except ClientError as e:
                print(f"Error uploading {file_path} to bucket {BUCKET_NAME}: {e}")
                return False

        else:
            print(f"Provided path {file_path} is neither a file nor a directory.")
            return False

class Delete:

    def __init__(self):
        self.client = boto3.client('s3',
                                endpoint_url=MINIO_ENDPOINT_URL,
                                aws_access_key_id=MINIO_ACCESS_KEY,
                                aws_secret_access_key=MINIO_SECRET_KEY)

    def delete_object(self,key: str) -> bool:
        """
        Delete a single object by key.
        """
        try:
            self.client.delete_object(Bucket=BUCKET_NAME, Key=key)
            logger.info(f"Deleted object {key} from bucket {BUCKET_NAME}")
            return True
        except ClientError as e:
            logger.error(f"Error deleting object {key} from bucket {BUCKET_NAME}: {e}")
            return False

if __name__ == "__main__":
    read = Read()
    read.download_object("indoor")