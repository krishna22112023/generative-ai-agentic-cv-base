import os
import logging
from typing import List, Optional

from langchain_core.tools import tool
from .decorators import log_io
from src.utils.optimize_preprocess import run_bayesian_optimization

logger = logging.getLogger(__name__)

@tool("preprocessing_pipeline")
@log_io
def preprocessing_pipeline(
    auto: bool = True,
    custom_pipeline: Optional[List[str]] = None,
    n_init: int = 20,
    n_iter: int = 30,
    q: int = 4,
) -> str:
    """Optimise or apply an image preprocessing pipeline.

    Parameters
    ----------
    auto : bool, default True
        When True, Bayesian optimisation is run to discover the best pipeline.
        When False, the provided *custom_pipeline* (or a default one) is used
        directly without optimisation.
    custom_pipeline : list[str] | None
        Ordered list of preprocessing function names. Ignored when *auto* is
        True. If None and *auto* is False, a sensible default is used.
    n_init : int, default 20
        Number of initial samples for Bayesian optimisation (only when auto).
    n_iter : int, default 30
        Number of optimisation iterations (only when auto).
    q : int, default 4
        Batch size for acquisition function during optimisation (only when auto).
    """
    DATA_DIR = os.getenv("DATA_DIR")
    input_path = Path(DATA_DIR) / "raw"
    artefacts_path = Path(DATA_DIR) / "artefacts"
    os.makedirs(artefacts_path,exist_ok=True)
    pipelines_path = artefacts_path / "pipelines"
    os.makedirs(pipelines_path,exist_ok=True)

    run_bayesian_optimization(
        auto=auto,
        custom_pipeline=custom_pipeline,
        n_init=n_init,
        n_iter=n_iter,
        q=q,
        input_path=input_path,
        pipelines_path=pipelines_path
    )

    mode = "auto-optimised" if auto else "custom/manual"
    return f"Preprocessing pipeline routine completed successfully ({mode} mode)."
