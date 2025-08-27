import argparse
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import torch
import pyiqa

logger = logging.getLogger(__name__)


class _ImagePath(BaseModel):
    image_path: str

def qalign_score(image_path: str) -> float:
    """Compute Q-Align quality score for *image_path* using *pyiqa*.

    The score roughly corresponds to the mean-opinion-score (MOS) predicted by
    the Q-Align paper (range ≈ 0–5 – higher is better).
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    iqa_metric = pyiqa.create_metric('qalign', device=device)
    return float(iqa_metric(image_path).item())

def create_app() -> FastAPI:
    app = FastAPI(title="Q-Align IQA API", description="Exposes qalign_score as an HTTP endpoint")

    @app.post("/qalign_score")
    async def qalign_score_endpoint(payload: _ImagePath):
        img_path = Path(payload.image_path)
        if not img_path.exists():
            raise HTTPException(status_code=400, detail=f"Image not found: {img_path}")
        try:
            score = qalign_score(str(img_path))
        except Exception as e:
            logger.exception("qalign_score failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e))
        return {"score": score}

    return app


def _parse_args():
    parser = argparse.ArgumentParser(description="Serve qalign_score via FastAPI")
    parser.add_argument("--host", default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5003, help="Port to bind (default: 5003)")
    parser.add_argument("--reload", action="store_true", help="Enable live reload (uvicorn --reload)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port, reload=args.reload)
