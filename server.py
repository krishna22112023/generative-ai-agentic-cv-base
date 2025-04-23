"""
Server script for running the API.
"""
import os
import logging
import uvicorn
import pyprojroot
import sys

root = pyprojroot.find_root(pyprojroot.has_dir("src"))
sys.path.append(str(root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting API server")
    logger.info("Creating default directories for data")
    for i in ["raw","processed","annotated","artefacts"]:
        os.makedirs(os.path.join(root,"data",i),exist_ok=True)
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
