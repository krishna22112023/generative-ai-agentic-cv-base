import os
import boto3
import logging
from typing import Dict, Any, List
from botocore.exceptions import ClientError

from src.config import MINIO_ENDPOINT_URL,MINIO_ACCESS_KEY,MINIO_SECRET_KEY,BUCKET_NAME

import pyprojroot
root = pyprojroot.find_root(pyprojroot.has_dir("src"))

logger = logging.getLogger(__name__)

class Read:

    def __init__(self):
        self.client = boto3.client('s3',
                                endpoint_url=MINIO_ENDPOINT_URL,
                                aws_access_key_id=MINIO_ACCESS_KEY,
                                aws_secret_access_key=MINIO_SECRET_KEY)

    def list_object(self, input_path: str) -> List[Dict[str, Any]]:
        """
        List objects in the bucket. Use the input_path to emulate folder listing.
        """
        logger.info(f"listing objects in minio bucket with prefix : {input_path}")
        try:
            response = self.client.list_objects_v2(
                Bucket=BUCKET_NAME, 
                Prefix=input_path, 
                MaxKeys=1000
            )
            objects = response.get('Contents', [])
            logger.info(f"Found {len(objects)} objects with input_path '{input_path}' in bucket {BUCKET_NAME}")
            return objects
        except ClientError as e:
            logger.error(f"Error listing objects with input_path '{input_path}' in bucket {BUCKET_NAME}: {e}")
            return []

    def download_object(self, input_path: str, output_path: str) -> bool:
        """_summary_

        Args:
            input_path (str): path to minio bucket folder
            output_path (str): path to local directory where the data will be downloaded

        Returns:
            bool: whether the download was successful (True/False)
        """
        objects = self.list_object(input_path)
        if not objects:
            logger.error(f"No objects found with input_path '{input_path}'.")
            return False

        # Treat the input_path as a directory and download all objects.
        # The object key includes the prefix like so : prefix_name/file_name.extension
        overall_success = True
        for obj in objects:
            key = obj.get("Key")

            # Skip "directory" placeholders (keys that end with "/").
            if key.endswith("/"):
                logger.debug("Skipping prefix placeholder %s", key)
                continue

            logger.info("minio object key found %s", key)

            local_file_path = os.path.join(output_path, key)
            logger.info(
                "creating a directory if does not exist %s", os.path.dirname(local_file_path)
            )
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            try:
                self.client.download_file(
                    Bucket=BUCKET_NAME, Key=key, Filename=local_file_path
                )
                logger.info("Downloaded %s to %s", key, output_path)
            except ClientError as e:
                logger.error("Error downloading %s to %s: %s", key, output_path, e)
                overall_success = False

        return overall_success

class Create:

    def __init__(self):
        self.client = boto3.client('s3',
                                endpoint_url=MINIO_ENDPOINT_URL,
                                aws_access_key_id=MINIO_ACCESS_KEY,
                                aws_secret_access_key=MINIO_SECRET_KEY)

    def upload_object(self, input_path: str, output_path: str) -> bool:
        """_summary_

        Args:
            input_path (str): path to local directory where the data will be uploaded
            output_path (str): path to minio folder in the bucket

        Returns:
            bool: whether the upload was successfull (True/False)
        """
        # If input_path is a directory, iterate through its files recursively.
        if os.path.isdir(input_path):
            all_success = True
            for root_dir, dirs, files in os.walk(input_path):
                for file in files:
                    full_file_path = os.path.join(root_dir, file)
                    # Obtain the relative path of the file with respect to input_path
                    rel_path = os.path.relpath(full_file_path, start=input_path)
                    # Construct the destination key using the provided key as the base
                    dest_key = os.path.join(output_path, rel_path).replace("\\", "/")
                    try:
                        self.client.upload_file(Filename=full_file_path, Bucket=BUCKET_NAME, Key=dest_key)
                        logger.info(f"Uploaded {full_file_path} to bucket {BUCKET_NAME} as {dest_key}")
                    except ClientError as e:
                        logger.info(f"Error uploading {full_file_path} to bucket {BUCKET_NAME}: {e}")
                        all_success = False
            return all_success

        # If file_path is a file, upload it directly.
        elif os.path.isfile(input_path):
            try:
                self.client.upload_file(Filename=input_path, Bucket=BUCKET_NAME, Key=output_path)
                logger.info(f"Uploaded {input_path} to bucket {BUCKET_NAME} as {output_path}")
                return True
            except ClientError as e:
                print(f"Error uploading {input_path} to bucket {BUCKET_NAME}: {e}")
                return False

        else:
            print(f"Provided path {input_path} is neither a file nor a directory.")
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
    input_path = "C:/Users/srikr/workspace/generative-ai-agentic-cv-base/data/Cats/artefacts"
    output_path = "Cats/artefacts"

    read = Create()
    read.upload_object(input_path, output_path)