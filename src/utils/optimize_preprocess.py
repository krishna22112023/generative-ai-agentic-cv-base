"""Optimization of preprocessing pipelines using Bayesian Optimization (BoTorch).

This module performs the following high-level steps for each *severity* bucket
("medium", "high", "very high"):

1. Build the image batch by filtering the BRISQUE scores saved by the IQA step
   (``artefacts/nr_iqa_results_raw.json``).
2. Define a pipeline search space (choice of 3 ordered preprocessing functions
   drawn from :pymod:`src.utils.preprocess` **and** their hyper-parameters).
3. Optimise the pipeline using Monte-Carlo Bayesian optimisation (BoTorch)
   to minimise the *mean* BRISQUE score over the batch.
4. Persist the best found pipeline + parameters to
   ``artefacts/pipelines/<severity>.json``.

Notes
-----
• CUDA is automatically used for GP modelling if available.
• Image evaluations are executed in parallel using ``concurrent.futures``.
• The design purposefully keeps the search space lean, it can be extended by
  adding more hyper-parameters or stages.
"""
from __future__ import annotations

import json
import os
import logging
import random
import shutil
import pyprojroot
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll

from src.utils import preprocess as pp
root = pyprojroot.find_root(pyprojroot.has_dir("src"))

logger = logging.getLogger(__name__)

# Map function index → callable + default params -------------------------------------------------
PREPROCESS_FUNCS: List[Tuple[str, callable]] = [
    ("blur", pp.blur),                # 0 – smoothing
    ("gaussian_blur", pp.gaussian_blur),  # 1 – smoothing
    ("median_blur", pp.median_blur),      # 2 – smoothing
    ("bilateral_filter", pp.bilateral_filter),  # 3 – smoothing
    (
        "fast_nl_means_denoising_colored",
        pp.fast_nl_means_denoising_colored,
    ),  # 4 – smoothing
    ("equalize_hist", pp.equalize_hist),  # 5 – contrast
    ("clahe_equalize", pp.clahe_equalize),  # 6 – contrast
    ("convert_scale_abs", pp.convert_scale_abs),  # 7 – brightness
    ("gamma_correction", pp.gamma_correction),  # 8 – brightness
]
# index 9 will be treated as a no-op (stage dropped)
FUNC_INDEX = {i: name for i, (name, _) in enumerate(PREPROCESS_FUNCS)}
FUNC_INDEX[9] = "noop"
# Reverse lookup name -> index (excludes noop)
NAME_TO_INDEX = {name: idx for idx, name in FUNC_INDEX.items()}
NAME_TO_INDEX.pop("noop", None)

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------


def severity_of(score: float) -> str | None:
    if score <= 20:
        return "very low"
    elif score > 20 and score < 40:
        return "low"
    elif score > 40 and score < 60:
        return "medium"
    elif score > 60 and score < 80:
        return "high"
    elif score > 80 and score <= 100:
        return "very high"
    return None


def load_batches(artefacts_path: Path, input_path: Path) -> Dict[str, List[Path]]:
    """Return mapping severity → list[Path] (only medium & worse)."""
    with open(artefacts_path / Path("nr_iqa_results_raw.json"), "r") as f:
        raw_scores = json.load(f)
    batches= {"medium": [], "high": [], "very_high": []}
    for fname, metrics in raw_scores.items():
        brisque = metrics.get("brisque")
        if brisque is None:
            continue
        sev = severity_of(brisque)
        if sev in {"medium", "high", "very high"}:
            key = sev.replace(" ", "_")  # "very high" → "very_high"
            batches[key].append(input_path / Path(fname))
    return batches


# -----------------------------------------------------------------------------
# Pipeline application helpers
# -----------------------------------------------------------------------------


def _apply_stage(image: np.ndarray, func_idx: int, params: Dict):
    """Apply a single preprocessing stage. Index 9 is a noop."""
    if func_idx == 9:  # noop – return image untouched
        return image

    # Safe access – indices 0-8 are valid
    try:
        name, fn = PREPROCESS_FUNCS[func_idx]
    except IndexError as e:
        logger.error("Invalid func_idx %s – out of PREPROCESS_FUNCS range", func_idx)
        raise

    # Dispatch parameters according to function signature
    if name in {"blur", "gaussian_blur"}:  # square kernel filters
        k = params["kernel_size"]
        return fn(image, ksize=(k, k))
    if name == "median_blur":
        k = params["kernel_size"]
        return fn(image, ksize=k)
    if name == "bilateral_filter":
        d = params["diameter"]
        return fn(image, d=d, sigmaColor=75, sigmaSpace=75)
    if name == "fast_nl_means_denoising_colored":
        h = params["luminance"]
        return fn(image, h=h, hColor=h)
    if name == "equalize_hist":
        return fn(image)
    if name == "clahe_equalize":
        return fn(image, clipLimit=params["clahe_clip"], tileGridSize=(8, 8))
    if name == "convert_scale_abs":
        return fn(image, alpha=params["alpha"], beta=0.0)
    if name == "gamma_correction":
        return fn(image, gamma=params["gamma"])

    raise ValueError(f"Unknown preprocessing function idx {func_idx}")


def apply_pipeline(image_path: Path, func_indices: Tuple[int, int, int], params: Dict):
    """Load image, sequentially apply stages, return processed image array."""
    img = cv2.imread(str(image_path))
    for idx in func_indices:
        if idx < 0:
            continue  # -1 represents no-op stage
        img = _apply_stage(img, idx, params)
    return img


# -----------------------------------------------------------------------------
# Objective evaluation
# -----------------------------------------------------------------------------


def brisque_score(img: np.ndarray, brisque_model) -> float:
    """Compute BRISQUE score of an image using the provided model."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    val = brisque_model.compute(gray)
    return float(val[0] if isinstance(val, (list, tuple)) else val)


def evaluate_candidate(
    func_indices: Tuple[int, int, int],
    params: Dict,
    images: List[Path],
    brisque_model,
    max_workers: int = 8,
) -> float:
    """Return **mean** BRISQUE given pipeline on ``images``."""

    # Human‐readable pipeline description
    pipeline_names = [FUNC_INDEX[i] for i in func_indices if i != 9]
    #print("Evaluating pipeline %s with params %s on %d images", pipeline_names, params, len(images))

    scores: List[float] = []

    def _worker(path):
        processed = apply_pipeline(path, func_indices, params)
        return brisque_score(processed, brisque_model)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_worker, p) for p in images]
        for fut in as_completed(futures):
            scores.append(fut.result())
    return float(np.mean(scores) if scores else 1e6)


# -----------------------------------------------------------------------------
# Bayesian Optimisation routine
# -----------------------------------------------------------------------------


def optimise_severity(
    severity: str,
    images: List[Path],
    n_init: int = 20,
    n_iter: int = 30,
    q: int = 4,
    pipelines_path: Path | None = None,
    brisque_model=None,
    processed_path: Path | None = None,
):
    """Run BO and save best pipeline for a severity bucket.
    Args:
        severity: str, the severity bucket to optimise
        images: List[Path], the images to optimise
        n_init: int, the number of initial points to sample
        n_iter: int, the number of iterations to run
        q: int, the number of points to sample in each iteration
    """
    if not images:
        logger.warning(f"[WARN] No images for severity '{severity}', skipping optimisation")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    dtype = torch.double

    # Variable ordering: stage1, stage2, stage3, smoothing_param, clahe_clip, alpha, gamma
    bounds = torch.tensor(
        [
            [0, 0, 0, 3.0, 2.0, 0.5, 0.4],   # lower bounds
            [9, 9, 9, 15.0, 40.0, 2.0, 2.5], # upper bounds (9 → noop)
        ],
        device=device,
        dtype=dtype,
    )

    # Severity-specific adjustments
    if severity == "medium":
        bounds[0, 3] = 0.5
        bounds[1, 3] = 2.0
    elif severity == "high":
        bounds[0, 3] = 1.0
        bounds[1, 3] = 3.0
    elif severity == "very_high":
        bounds[0, 3] = 2.0
        bounds[1, 3] = 3.0

    def decode_x(x: torch.Tensor):
        x = x.cpu()
        s1, s2, s3, smooth_p, clip, alpha, gamma = x.tolist()
        func_ids = tuple(int(round(v)) for v in (s1, s2, s3))
        # ensure integer within [0,9]
        func_ids = tuple(max(0, min(9, fid)) for fid in func_ids)
        # sanitise smoothing param to nearest odd integer within [3,15]
        k = int(round(smooth_p))
        if k % 2 == 0:
            k += 1 if k < 15 else -1
        param_dict = {
            "kernel_size": k,
            "diameter": k,
            "luminance": k,
            "clahe_clip": clip,
            "alpha": alpha,
            "gamma": gamma,
        }
        return func_ids, param_dict

    # Sobol initial design ------------------------------------------------------
    sobol = torch.quasirandom.SobolEngine(dimension=bounds.shape[1], scramble=True)
    X_raw = bounds[0] + (bounds[1] - bounds[0]) * sobol.draw(n_init).to(device).double()

    # Min-max normalise to unit cube for GP fitting
    def _norm(x):
        return (x - bounds[0]) / (bounds[1] - bounds[0])

    X = _norm(X_raw)

    Y_list = []
    for i in range(n_init):
        funcs, params = decode_x(X_raw[i])
        score = evaluate_candidate(funcs, params, images, brisque_model)
        Y_list.append([score])
    Y = torch.tensor(Y_list, device=device, dtype=dtype)

    best_index = torch.argmin(Y)
    best_x = X_raw[best_index].detach().cpu()
    best_y = Y[best_index].item()

    # BO loop -------------------------------------------------------------------
    for iteration in range(n_iter):
        # Fit GP
        gp = SingleTaskGP(X, standardize(Y))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        gp = gp.to(device)
        mll = mll.to(device)

        # Optimise hyper-parameters
        fit_gpytorch_mll(mll)

        # Acquisition
        sampler = SobolQMCNormalSampler(torch.Size([128]))
        qEI = qLogExpectedImprovement(model=gp, best_f=Y.min(), sampler=sampler)

        candidate, _ = optimize_acqf(
            acq_function=qEI,
            bounds=torch.stack([torch.zeros_like(bounds[0]), torch.ones_like(bounds[1])]).to(device),
            q=q,
            num_restarts=10,
            raw_samples=256,
        )

        # candidate is in unit cube → de-normalise to real parameter space
        cand_raw = candidate[0]  # take first element if q>1
        cand_denorm = bounds[0] + (bounds[1] - bounds[0]) * cand_raw
        funcs, params = decode_x(cand_denorm)
        score = evaluate_candidate(funcs, params, images, brisque_model)

        # Update raw and normalised design points
        X_raw = torch.cat([X_raw, cand_denorm.unsqueeze(0)], dim=0)
        X = torch.cat([X, _norm(cand_denorm).unsqueeze(0)], dim=0)
        Y = torch.cat([Y, torch.tensor([[score]], device=device, dtype=dtype)], dim=0)

        if score < best_y:
            best_y = score
            best_x = cand_denorm.detach().cpu()
        print(
            f"[{severity}] Iter {iteration+1}/{n_iter}-> best mean BRISQUE so far: {best_y:.3f}"
        )

    # Save best pipeline --------------------------------------------------------
    result_list = []
    best_funcs, best_params = decode_x(best_x)
    result = {
        "functions": [FUNC_INDEX[i] for i in best_funcs if i != 9],
        "params": best_params,
        "score": best_y,
    }
    out_path = pipelines_path / f"{severity}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    logger.info(f"[{severity}] optimisation finished -> saved to {out_path}")
    result_list.append(result)

    if processed_path is not None:
        save_dir = Path(processed_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        for img_path in images:
            processed = apply_pipeline(img_path, best_funcs, best_params)
            out_file = save_dir / img_path.name
            cv2.imwrite(str(out_file), processed)
        logger.info("[%s] Saved %d processed images to %s", severity, len(images), save_dir)
    return result_list

def run_bayesian_optimization(
    auto: bool = True,
    custom_pipeline: List[str] | None = None,
    n_init: int = 20,
    n_iter: int = 30,
    q: int = 4,
    input_path: Path = None,
    artefacts_path: Path = None,
    processed_path: Path = None,
    ):
    """Optimise preprocessing pipelines or apply a user-defined one.

    Parameters
    ----------
    auto : bool, default True
        If True, Bayesian optimisation is executed to find the best pipeline.
        When False, the provided *custom_pipeline* (or a default) is evaluated
        and persisted without optimisation.
    custom_pipeline : list[str] | None
        Ordered list of preprocessing function names. Only used when
        ``auto`` is False. Unknown names are ignored.
    n_init : int, default 20
        Number of initial Sobol samples. Only relevant when ``auto`` is True.
    n_iter : int, default 30
        Number of optimisation iterations. Only relevant when ``auto`` is True.
    q : int, default 4
        Batch size for qEI acquisition. Only relevant when ``auto`` is True.
    """

    # Lazily instantiate the BRISQUE model to avoid heavyweight global initialisation
    BRISQUE_MODEL = cv2.quality.QualityBRISQUE_create(
        str(root / "src" / "config" / "brisque" / "brisque_model_live.yml"),
        str(root / "src" / "config" / "brisque" / "brisque_range_live.yml"),
    )
    batches = load_batches(artefacts_path,input_path)
    logger.info(f"Batches created from {input_path}")
    pipelines_path = artefacts_path / Path("pipelines")

    results: List[Dict] = []  # collect results for return

    if auto:
        # ------------------------- Bayesian optimisation path -------------------------
        with ThreadPoolExecutor(max_workers=3) as ex:
            futures = {
                ex.submit(
                    optimise_severity,
                    sev,
                    imgs,
                    n_init,
                    n_iter,
                    q,
                    pipelines_path,
                    BRISQUE_MODEL,
                    processed_path,
                ): sev
                for sev, imgs in batches.items()
                if imgs
            }
            for fut in as_completed(futures):
                sev = futures[fut]
                try:
                    res = fut.result()
                    if res:  # extend with list of dicts
                        results.extend(res)
                except Exception as e:
                    logger.error(f"[ERROR] optimisation for severity '{sev}' failed: {e}")
    else:
        if not custom_pipeline:
            # Sensible default if user does not provide anything
            custom_pipeline = [
                "median_blur",
                "clahe_equalize",
                "gamma_correction",
            ]

        # Map function names to indices, pad / truncate to exactly 3 stages
        func_ids = [NAME_TO_INDEX.get(name, 9) for name in custom_pipeline[:3]]
        while len(func_ids) < 3:
            func_ids.append(9)  # pad with noop
        func_indices = tuple(func_ids)  # type: ignore

        # Default parameters (can be expanded later)
        params = {"kernel_size": 3, "diameter": 3, "luminance": 3, "clahe_clip": 2.0, "alpha": 1.0, "gamma": 1.0}

        for sev, imgs in batches.items():
            if not imgs:
                continue
            score = evaluate_candidate(func_indices, params, imgs, BRISQUE_MODEL)
            result = {
                "functions": [FUNC_INDEX[i] for i in func_indices if i != 9],
                "params": params,
                "score": score,
                "severity": sev,
            }
            results.append(result)
            out_path = pipelines_path / f"{sev}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=4)
            logger.info(
                f"[{sev}] manual pipeline evaluated -> mean BRISQUE: {score:.3f}, saved to {out_path}"
            )

            # Save processed images under manual path
            if processed_path is not None:
                save_dir = Path(processed_path) / sev
                save_dir.mkdir(parents=True, exist_ok=True)
                for img_path in imgs:
                    processed_img = apply_pipeline(img_path, func_indices, params)
                    cv2.imwrite(str(save_dir / img_path.name), processed_img)
                logger.info("[%s] Saved %d processed images to %s (manual pipeline)", sev, len(imgs), save_dir)

    return results


if __name__ == "__main__":
    run_bayesian_optimization(input_path=str(Path("C:/Users/srikr/workspace/generative-ai-agentic-cv-base/data/Cats/raw")), artefacts_path=str(Path("C:/Users/srikr/workspace/generative-ai-agentic-cv-base/data/Cats/artefacts/")),processed_path=str(Path("C:/Users/srikr/workspace/generative-ai-agentic-cv-base/data/Cats/processed/")))
