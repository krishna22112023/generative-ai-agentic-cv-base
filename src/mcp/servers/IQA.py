from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder, SystemMessagePromptTemplate
)
from pydantic import BaseModel, Field
from typing import List
import logging
from mcp.server.fastmcp import FastMCP
import sys
from PIL import Image
import io
import os
import base64
import glob
import json
from pathlib import Path
import pyprojroot

from src.agents.llm import get_llm_by_type
from decorators import log_io

root = pyprojroot.find_root(pyprojroot.has_dir("src"))
sys.path.append(str(root))
logger = logging.getLogger(__name__)

mcp = FastMCP("IQA")

class Degradation(BaseModel):
    degradation: str = Field(..., description="One of the seven degradation types.")
    thought: str = Field(..., description="The assessment thought for the degradation.")
    severity: str = Field(..., description="Severity rating (very low, low, medium, high, very high).")

class IQAResponse(BaseModel):
    items: List[Degradation]

def resize_image(image_path, max_size=(512, 512)):
    with Image.open(image_path) as img:
        img.thumbnail(max_size)
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        return buf.getvalue()

LOCAL_DIR = "data/raw"

@mcp.tool()
@log_io
def openai_vlm_iqa(prefix: str) -> str:
    """
    Assess the quality of images in a directory using a pre-trained model based on 7 degredations that include 
    noise, motion blur, defocus blur, haze, rain, dark, and jpeg compression artifact and classify them into 5 severity levels 
    namely "very low", "low", "medium", "high", and "very high".
    Infer the prefix from the user's request. The user can either specify prefix of files in minIO or local file system.
    """
    llm = get_llm_by_type("vision")

    # Inject the output parser’s instructions into your system prompt.
    fp_prompt = f"{root}/src/prompts/openai_iqa.md"
    with open(fp_prompt, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    system_template = SystemMessagePromptTemplate.from_template(prompt_template)
    # system_template = hub.pull(agent_config.LANGSMITH_PROMPT_TEMPLATE_NAME)

    messages = MessagesPlaceholder(variable_name='messages')
    prompt = ChatPromptTemplate.from_messages([system_template, messages])
    chain = prompt | llm.with_structured_output(schema=IQAResponse)

    results = {}
    image_files = glob.glob(f"{root}/{LOCAL_DIR}/{prefix}/*.jpg")
    total = len(image_files)

    logger.info({'status': f"Starting Image Quality Assessment on {total} images"})

    for file in image_files:
        image_data = resize_image(file)
        encoded_data = base64.b64encode(image_data).decode('utf-8')
        message = [{"role": "user", "content": f"data:image/jpeg;base64,{encoded_data}"}]
        response = chain.invoke({"messages": message})
        try:
            parsed = response.model_dump()["items"]
            # Filter out degradations with severity "high" or "very high"
            #filtered = [item for item in parsed if item.get("severity") in ("high", "very high")]
            results[f"{LOCAL_DIR}/{prefix}/{Path(file).name}"] = parsed
        except json.JSONDecodeError as e:
            results[f"{LOCAL_DIR}/{prefix}/{Path(file).name}"] = {"error": "JSON decode error", "detail": str(e)}
        
    logger.info({'status': f"Summarizing image quality results on {total} images. Showing top 5 only."})
            
    # Save raw results to image_dir/intermediate_results/iqa_results.json
    intermediate_dir = os.path.join(root,"data","intermediate_results",prefix)
    os.makedirs(intermediate_dir, exist_ok=True)
    intermediate_path = os.path.join(intermediate_dir, "iqa_results.json")
    with open(intermediate_path, 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, indent=4)
    
    # Create aggregated results table.
    # Columns are degradation types; rows are severity levels.
    degradations = ["noise", "motion blur", "defocus blur", "haze", "rain", "dark", "jpeg compression artifact"]
    severities = ["very low", "low", "medium","high", "very high"]
    aggregated = {sev: {deg: 0 for deg in degradations} for sev in severities}
    
    for items in results.values():
        if isinstance(items, list):
            for item in items:
                sev = item.get("severity")
                deg = item.get("degradation")
                if sev in severities and deg in degradations:
                    aggregated[sev][deg] += 1
    with open(os.path.join(intermediate_dir, "iqa_results_aggregated.json"), 'w', encoding='utf-8') as outfile:
        json.dump(aggregated, outfile, indent=4)

    return json.dumps(aggregated, indent=4)
    
if __name__ == "__main__":
    mcp.run(transport="stdio")
