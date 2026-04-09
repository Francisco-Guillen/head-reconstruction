import shutil
import subprocess
from pathlib import Path
from pipeline.base import Stage


class SfMStage(Stage):
    """Structure-from-Motion using COLMAP."""

    def __init__(self, quality: str = "medium", use_masks: bool = False):
        super().__init__("SfM")
        self.quality = quality
        self.use_masks = use_masks

    def _run(self, cmd: list, label: str):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stderr[-2000:])
            raise RuntimeError(f"{label} failed")

    def run(self, context: dict) -> dict:
        input_dir = Path(context.get("sfm_input_dir", context["input_dir"]))
        work_dir = Path(context["work_dir"])

        database = work_dir / "database.db"
        sparse_dir = work_dir / "sparse"

        if database.exists():
            database.unlink()
        if sparse_dir.exists():
            shutil.rmtree(sparse_dir)
        sparse_dir.mkdir(parents=True, exist_ok=True)

        if self.quality == "low":
            max_image_size, max_num_features = "800", "2048"
        elif self.quality == "high":
            max_image_size, max_num_features = "1200", "8192"
        else:
            max_image_size, max_num_features = "1000", "4096"

        use_gpu = "0" if context.get("no_gpu", False) else "1"

        cmd = [
            "colmap", "feature_extractor",
            "--database_path", str(database),
            "--image_path", str(input_dir),
            "--ImageReader.single_camera", "1",
            "--FeatureExtraction.use_gpu", use_gpu,
            "--FeatureExtraction.max_image_size", max_image_size,
            "--SiftExtraction.max_num_features", max_num_features,
        ]

        mask_dir = context.get("mask_dir")
        if self.use_masks and mask_dir:
            cmd += ["--ImageReader.mask_path", str(mask_dir)]

        self._run(cmd, "Feature extraction")
        self._run([
            "colmap", "exhaustive_matcher",
            "--database_path", str(database),
            "--FeatureMatching.use_gpu", use_gpu,
        ], "Feature matching")
        self._run([
            "colmap", "mapper",
            "--database_path", str(database),
            "--image_path", str(input_dir),
            "--output_path", str(sparse_dir),
        ], "Sparse reconstruction")

        model_dir = sparse_dir / "0"
        if not model_dir.exists():
            raise RuntimeError("No sparse model generated")

        context["database"] = str(database)
        context["sparse_model_dir"] = str(model_dir)
        return context
