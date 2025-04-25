import os
from dotenv import load_dotenv
import pyprojroot

# Load environment variables
root = pyprojroot.find_root(pyprojroot.has_dir("src"))
load_dotenv(os.path.join(root,".env"))

# Tool configuration
IQA_API_KEY= os.getenv("BASIC_API_KEY") #for image quality analysis
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY= os.getenv("MINIO_SECRET_KEY")
MINIO_ENDPOINT_URL= os.getenv("MINIO_ENDPOINT_URL")
BUCKET_NAME=os.getenv("BUCKET_NAME")

# Preprocessor configuration
ABS_PATH_TO_PYTHON_ENV = "/Users/krishnaiyer/miniforge3/envs/restormer/bin/python"

# Annotation configuration
ANNOTATION_MODEL = os.getenv("ANNOTATION_MODEL", "gemini-2.5-pro-exp-03-25")
ANNOTATION_API_KEY = os.getenv("ANNOTATION_API_KEY")

MCP_TOOL_MAP: dict[str, list] = {
    "coordinator": [],
    "supervisor": [], 
    "data_collector": ["minio"],  
    "data_quality": ["IQA"],  
    "data_preprocessor": ["IR"],  
    "data_annotator": ["annotator"],
    "reporter":[]
}

USE_MCP = False

PREPROCESSOR_MODEL_MAP: dict[str, dict] = {
    "restormer":
        {
        "noise": "Real_Denoising",
        "motion blur": "Single_Image_Defocus_Deblurring",
        "defocus blur": "Motion_Deblurring",
        "rain": "Deraining",
        }   
    }