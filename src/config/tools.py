import os
from dotenv import load_dotenv
import pyprojroot

# Load environment variables
root = pyprojroot.find_root(pyprojroot.has_dir("config"))
load_dotenv(os.path.join(root,".env"))

# Tool configuration
IQA_API_KEY= os.getenv("BASIC_API_KEY") #for image quality analysis
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY= os.getenv("MINIO_SECRET_KEY")
MINIO_ENDPOINT_URL= os.getenv("MINIO_ENDPOINT_URL")
BUCKET_NAME=os.getenv("BUCKET_NAME")
