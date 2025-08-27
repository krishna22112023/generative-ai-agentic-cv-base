"""AutoRestore
=============================

This module provides the *Planning* class which orchestrates the first phase of
AutoRestore: analysing degradations, batching images, and exhaustively
searching for an optimal restoration pipeline using the Qwen-Image-Edit model
and Q-Align scores.

The class is intentionally **self-contained** – it writes intermediate artefacts
(JSON descriptions and temporary images) that downstream stages
(Execution/Verification) can consume.
"""

from __future__ import annotations

from pathlib import Path
from itertools import permutations
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import json
import logging
import os
import random
import shutil
from typing import Dict, List, Tuple
import pyprojroot
import sys

root = pyprojroot.find_root(pyprojroot.has_dir("src"))
sys.path.append(str(root))

from src.utils.IQA import DepictQA, QAlign 
from src.utils.preprocess import qwen_preprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _ensure_dir(path: Path) -> None:
    """Create *path* recursively if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def _write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class Planning:
    """Implements the *Planning* stage of AutoRestore.

    Parameters
    ----------
    data_path : str | Path
        Directory that contains raw input images.
    artefacts_path : str | Path
        Directory where IQA/batch/pipeline JSON files and temporary images will
        be written.
    num_workers : int, default=4
        Degree of parallelism when running *qwen_preprocess*.
    """

    SEVERITY_ORDER = ["very low", "low", "medium", "high", "very high"]
    SEVERITY_THRESHOLD = {"medium", "high", "very high"}

    def __init__(self, data_path: str | Path, artefacts_path: str | Path, *, num_workers: int = 4) -> None:
        self.data_path = Path(root / data_path)
        self.artefacts_path = Path(root / artefacts_path)
        self.num_workers = num_workers
        _ensure_dir(self.artefacts_path)
        self.depictqa = DepictQA()
        self.qalign = QAlign()

    def image_quality_analysis(
        self,
        images_subset: List[str] | None = None,
        *,
        replan: bool = False,
        previous_plans: Dict[str, str] | None = None,
    ) -> None:
        """Run DepictQA on every image and persist full results to *IQA.json*.

        The resulting JSON maps *filename* → list[(degradation, severity)]. No
        filtering is applied at this stage.
        """
        logger.info("Running image-quality analysis with DepictQA…")

        ext = ("*.jpg", "*.jpeg", "*.png")
        image_paths_all = [p for pattern in ext for p in self.data_path.glob(pattern)]
        if images_subset is not None:
            names_set = set(images_subset)
            image_paths = [p for p in image_paths_all if p.name in names_set]
        else:
            image_paths = image_paths_all
        if not image_paths:
            logger.warning("No images found for IQA – aborting planning stage.")
            return

        per_image: Dict[str, List[Tuple[str, str]]] = {}

        for img_path in image_paths:
            try:
                prev_plan_str = None
                if replan and previous_plans is not None:
                    prev_plan_str = previous_plans.get(img_path.name)
                # DepictQA returns (prompt, list[(degradation, level), …])
                _, res = self.depictqa.query(
                    [img_path],
                    "eval_degradation",
                    replan=replan,
                    previous_plan=prev_plan_str,
                )
            except Exception as e:  
                logger.warning("DepictQA failed for %s: %s", img_path.name, e)
                res = []

            # Store full list – keep DepictQA wording intact
            per_image[img_path.name] = res

        if not per_image:
            logger.warning("No images with severity ≥ medium found – nothing to batch.")
            return

        # Persist IQA.json
        iqa_path = self.artefacts_path / "IQA.json"
        _write_json(per_image, iqa_path)
        logger.info("IQA.json written to %s", iqa_path)

        # Derive batches next
        self.batch_creation(per_image)

    def batch_creation(self, per_image: Dict[str, List[Tuple[str, str]]]) -> None:
        """Generate *batch_IQA.json* where keys are unique degradation combos (≤3)."""

        all_degradations = [
            "motion blur",
            "defocus blur",
            "rain",
            "raindrop",
            "haze",
            "dark",
            "noise",
            "jpeg compression artifact",
        ]

        from itertools import combinations

        def _key_from_combo(combo: Tuple[str, ...]) -> str:
            return "-".join(sorted(d.replace(" ", "_") for d in combo))

        # Initialise dictionary with *all* possible combos (92 keys)
        batch_dict: Dict[str, List[str]] = {
            _key_from_combo(c): []
            for r in (1, 2, 3)
            for c in combinations(all_degradations, r)
        }

        severity_rank = {"very high": 3, "high": 2, "medium": 1}

        for img_name, deg_list in per_image.items():
            # Filter degradations by severity ≥ medium
            valid: List[Tuple[str, str]] = [
                (d, s.lower()) for d, s in deg_list if s.lower() in severity_rank
            ]
            if not valid:
                continue  # skip images without significant degradation

            # Sort by severity rank desc, keep unique degradation names
            valid_sorted = sorted(valid, key=lambda x: -severity_rank[x[1]])
            unique_deg = []
            for d, _ in valid_sorted:
                if d not in unique_deg:
                    unique_deg.append(d)
                if len(unique_deg) == 3:
                    break

            key = _key_from_combo(tuple(unique_deg))
            batch_dict[key].append(img_name)

        out_path = self.artefacts_path / "batch_IQA.json"
        _write_json(batch_dict, out_path)
        logger.info("batch_IQA.json written to %s", out_path)

    def batch_process(self, *, num_workers: int | None = None, batch_process: bool = True) -> None:
        """Run pipeline search in parallel.

        Writes *batch_pipeline.json* containing the optimal sequence per
        degradation-ID.
        """
        if not batch_process:
            logger.info("batch_process flag is False – skipping exhaustive search.")
            return

        num_workers = num_workers or self.num_workers
        batch_file = self.artefacts_path / "batch_IQA.json"
        if not batch_file.exists():
            raise FileNotFoundError(f"{batch_file} not found. Run image_quality_analysis first.")

        batches: Dict[str, List[str]] = _read_json(batch_file)

        # Load IQA for severity lookup
        iqa_data: Dict[str, List[Tuple[str, str]]] = _read_json(self.artefacts_path / "IQA.json")

        tmp_dir = self.artefacts_path / "tmp_planning"
        _ensure_dir(tmp_dir)

        best_pipelines: Dict[str, List[str]] = {}

        # Build an iterator over batches in chunks of *num_workers*
        batch_items = list(batches.items())
        for i in range(0, len(batch_items), num_workers):
            chunk = batch_items[i : i + num_workers]
            futures = {}
            with ThreadPoolExecutor(max_workers=num_workers) as exe:
                for degradation_id, img_list in chunk:
                    # Filter images with at least one degradation ≥ medium (should already hold)
                    all_imgs = img_list
                    if not all_imgs:
                        continue
                    img_choice = Path(self.data_path / random.choice(all_imgs))
                    futures[exe.submit(self._search_best_pipeline, degradation_id, img_choice, iqa_data[img_choice.name])] = degradation_id

                for fut in as_completed(futures):
                    degradation_id = futures[fut]
                    try:
                        best_seq = fut.result()
                        if best_seq:
                            best_pipelines[degradation_id] = best_seq
                    except Exception as e:
                        logger.error("Pipeline search failed for %s: %s", degradation_id, e)

        # Persist results
        out_path = self.artefacts_path / "batch_pipeline.json"
        _write_json(best_pipelines, out_path)
        logger.info("batch_pipeline.json written to %s", out_path)

        # Cleanup
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def _search_best_pipeline(
        self,
        degradation_id: str,
        img_path: Path,
        img_deg_list: List[Tuple[str, str]],
    ) -> List[str] | None:
        """Enumerate all permutations of the degradation list and return the best sequence."""
        degradation_types = [d.replace("_", " ") for d in degradation_id.split("-")]
        best_score = float("-inf")
        best_sequence: List[str] | None = None

        # Build severity map from the chosen image's IQA list
        severity_map = {d: s for d, s in img_deg_list if d in degradation_types}
        severity_map = {d: severity_map.get(d, "medium") for d in degradation_types}

        for perm in permutations(degradation_types, len(degradation_types)):
            current_img_path = img_path
            for step_idx, degradation in enumerate(perm):
                severity = severity_map[degradation]
                tmp_out_dir = self.artefacts_path / "tmp_planning" / degradation_id / "_".join(perm)
                _ensure_dir(tmp_out_dir)
                processed = qwen_preprocess(
                    str(current_img_path),
                    (degradation, severity),
                    str(tmp_out_dir),
                )
                # qwen_preprocess returns response; the saved image path is constructed inside.
                # We reconstruct what that path is so we can feed it to next stage:
                next_img = tmp_out_dir / f"{current_img_path.name}-{degradation}-{severity}.jpg"
                current_img_path = next_img

            # Finished the pipeline – compute quality score
            try:
                score = self.qalign.query(str(current_img_path))
            except Exception:
                score = -float("inf")
            if score > best_score:
                best_score = score
                best_sequence = list(perm)

        return best_sequence

class Executor:
    """Apply the optimal pipelines discovered by :class:`Planning`.

    Parameters
    ----------
    data_path : str | Path
        Directory containing raw images.
    artefacts_path : str | Path
        Directory where *batch_pipeline.json*, *batch_IQA.json* and *IQA.json*
        reside.
    processed_path : str | Path
        Destination directory for final, restored images.
    num_workers : int, default=4
        Parallel workers used when invoking *qwen_preprocess*.
    model_type : str, default="qwen-edit"
        Placeholder for future extensibility – currently unused.
    """

    def __init__(
        self,
        data_path: str | Path,
        artefacts_path: str | Path,
        processed_path: str | Path,
        *,
        num_workers: int = 4,
        model_type: str = "qwen-edit",
        batch: bool = True,
    ) -> None:
        root_path = root  # from earlier pyprojroot lookup
        self.data_path = Path(root_path / data_path)
        self.artefacts_path = Path(root_path / artefacts_path)
        self.processed_path = Path(root_path / processed_path)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        self.num_workers = num_workers
        self.model_type = model_type
        self.batch = batch

        # Load artefacts
        self.batch_pipeline: Dict[str, List[str]] = _read_json(self.artefacts_path / "batch_pipeline.json")
        self.batch_iqa: Dict[str, List[str]] = _read_json(self.artefacts_path / "batch_IQA.json")
        self.iqa_data: Dict[str, List[Tuple[str, str]]] = _read_json(self.artefacts_path / "IQA.json")

    def run(self) -> None:
        """Execute restoration pipelines in parallel."""
        tmp_root = self.artefacts_path / "tmp_execute"
        _ensure_dir(tmp_root)

        tasks = []  # list of (img_name, degradation_seq)
        for combo_key, seq in self.batch_pipeline.items():
            imgs = self.batch_iqa.get(combo_key, [])
            if not imgs:
                continue
            for img_name in imgs:
                tasks.append((img_name, seq))

        if not tasks:
            logger.warning("Executor found no images to process.")
            return

        def _process(item):
            img_name, seq = item
            raw_path = self.data_path / img_name
            if not raw_path.exists():
                logger.error("Raw image missing: %s", raw_path)
                return

            # Build severity map for this image
            deg_list = self.iqa_data.get(img_name, [])
            severity_map = {d: s for d, s in deg_list}

            current_img_path = raw_path
            tmp_dir = tmp_root / img_name
            _ensure_dir(tmp_dir)
            for step_idx, degradation in enumerate(seq):
                # seq items are space-separated original names
                sev = severity_map.get(degradation, "medium")
                logger.info(f"Step {step_idx+1} of {len(seq)}: Processing {img_name} with degradation {degradation} and severity {sev}")
                out_dir = tmp_dir
                qwen_preprocess(str(current_img_path), (degradation, sev), str(out_dir))
                # Determine saved image path
                current_img_path = out_dir / f"{current_img_path.name}-{degradation}-{sev}.jpg"

            # Copy / move final image to processed_path with original filename
            final_path = self.processed_path / img_name
            try:
                shutil.move(str(current_img_path), final_path)
            except shutil.Error:
                # Different filesystem – fallback to copy
                shutil.copy(str(current_img_path), final_path)

            # Cleanup tmp
            shutil.rmtree(tmp_dir, ignore_errors=True)

        if self.batch:
            # Parallel execution
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=self.num_workers) as exe:
                futures = {exe.submit(_process, item): item for item in tasks}
                for fut in as_completed(futures):
                    exc = fut.exception()
                    if exc:
                        img_name, _ = futures[fut]
                        logger.error("Processing failed for %s: %s", img_name, exc)
        else:
            for item in tasks:
                try:
                    _process(item)
                except Exception as e:
                    img_name, _ = item
                    logger.error("Processing failed for %s: %s", img_name, e)

        # Cleanup root tmp dir
        shutil.rmtree(tmp_root, ignore_errors=True)


class Verifier:
    """Compare raw vs processed images using DepictQA."""

    def __init__(
        self,
        data_path: str | Path,
        processed_path: str | Path,
        artefacts_path: str | Path,
    ) -> None:
        base = root
        self.data_path = Path(base / data_path)
        self.processed_path = Path(base / processed_path)
        self.artefacts_path = Path(base / artefacts_path)
        _ensure_dir(self.artefacts_path)
        self.qalign = QAlign()

    def run(self) -> None:
        failed: List[str] = []
        verify_scores: Dict[str, Tuple[List, List]] = {}

        for proc_img in self.processed_path.glob("*.jpg"):
            logger.info(f"Verifying quality of {proc_img.name}")
            raw_img = self.data_path / proc_img.name
            if not raw_img.exists():
                logger.warning("Missing raw counterpart for %s", proc_img.name)
                continue

            try:
                # Individual degradation evaluation for record
                raw_score = self.qalign.query(str(raw_img))
                proc_score = self.qalign.query(str(proc_img))
                verify_scores[proc_img.name] = [raw_score, proc_score]
                logger.info(f"verified {proc_img.name} = raw image score: {raw_score}, processed image score: {proc_score}")
                if proc_score <= raw_score:
                    logger.info(f"verification failed for {proc_img.name}. appended to failed list to try again.")
                    failed.append(proc_img.name)
            except Exception as e:
                logger.error("DepictQA processing failed for %s: %s", proc_img.name, e)
            logger.info(f"verification successful for {proc_img.name}")

        # Save failed list
        failed_path = self.artefacts_path / "failed_IQA.json"
        _write_json(failed, failed_path)
        logger.info("Verification complete. %d failures saved to %s", len(failed), failed_path)

        # Save evaluation details
        eval_path = self.artefacts_path / "verify_IQA.json"
        _write_json(verify_scores, eval_path)
        logger.info("Per-image evaluation saved to %s", eval_path)


if __name__ == "__main__":
    data_path = Path("data/raw")
    artefacts_path = Path("data/artefacts")
    processed_path = Path("data/processed")
    max_retries = 3

    # Initialise raw.json if not present
    raw_list_path = artefacts_path / "raw.json"
    if not raw_list_path.exists():
        all_raw = [p.name for p in data_path.glob("*.jpg")]
        _write_json(all_raw, raw_list_path)

    for attempt in range(1, max_retries + 1):
        to_process: List[str] = _read_json(raw_list_path)
        if not to_process:
            logger.info("All images successfully restored after %d iteration(s).", attempt - 1)
            break

        logger.info("--- Iteration %d: processing %d image(s) ---", attempt, len(to_process))

        # Prepare previous plan mapping (if not first iteration)
        prev_plans_map: Dict[str, str] | None = None
        if attempt > 1:
            try:
                combo_to_plan = _read_json(artefacts_path / "batch_pipeline.json")
                combo_to_imgs = _read_json(artefacts_path / "batch_IQA.json")
                prev_plans_map = {}
                for combo, imgs in combo_to_imgs.items():
                    if combo in combo_to_plan:
                        seq = combo_to_plan[combo]
                        seq_str = " -> ".join(seq)
                        for img in imgs:
                            prev_plans_map[img] = seq_str
            except Exception:
                prev_plans_map = None

        # Planning restricted to failed/raw list
        planning = Planning(data_path, artefacts_path)
        planning.image_quality_analysis(images_subset=to_process, replan=(attempt > 1), previous_plans=prev_plans_map)
        planning.batch_process()

        # Execute
        executor = Executor(data_path, artefacts_path, processed_path, batch=False)
        executor.run()

        # Verify
        verifier = Verifier(data_path, processed_path, artefacts_path)
        verifier.run()

        failed_path = artefacts_path / "failed_IQA.json"
        failed_imgs: List[str] = _read_json(failed_path) if failed_path.exists() else []

        _write_json(failed_imgs, raw_list_path)

    else:
        logger.info("Maximum retries (%d) reached. Some images still failed.", max_retries)