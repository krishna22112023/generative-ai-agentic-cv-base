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
ANNOTATION_MODEL = os.getenv("ANNOTATION_MODEL", "grounded-sam")

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

PREPROCESSOR_MODEL_MAP: dict[str, dict[str, list[str]]] = {
    "restormer": {
        "noise": ["Real_Denoising","Gaussian_Color_Denoising"],
        "motion blur": ["Single_Image_Defocus_Deblurring"],
        "defocus blur": ["Motion_Deblurring"],
        "rain": ["Deraining"],
    },
    "swinir": {
        "noise": ["color_dn_15", "color_dn_25", "color_dn_50"],
        "jpeg compression": ["color_jpeg_car_10", "color_jpeg_car_20", "color_jpeg_car_30", "color_jpeg_car_40"],
        "poor resolution": ["real_sr"]
    },
    "xrestormer": {
        "noise": ["denoising"],
        "haze": ["dehazing"],
        "rain": ["deraining"],
        "poor resolution": ["super_resolution"],
        "motion blur": ["motion_deblurring"],
    }
}

MODEL_SCRIPT_CONFIG = {
    "restormer": {"env": "restormer", 
                  "python": "python3.8", 
                  "script": "Restormer/demo.py"},
    "swinir": {"env": "swinir", 
               "python": "python3.8", 
               "script": "SwinIR/inference.py"},
    "xrestormer": {"env": "xrestormer", 
                   "python": "python3.8", 
                   "script": "X-Restormer/inference.py"},
}


TAVILY_MAX_RESULTS = 5