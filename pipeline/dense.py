import shutil
import subprocess
from pathlib import Path
from pipeline.base import Stage


class DenseStage(Stage):
    """Dense reconstruction using COLMAP MVS."""

    def __init__(self, quality: str = "medium"):
        super().__init__("Dense")
        self.quality = quality

    def _run(self, cmd: list, label: str):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stderr[-2000:])
            raise RuntimeError(f"{label} failed")

    def run(self, context: dict) -> dict:
        input_dir = Path(context.get("sfm_input_dir", context["input_dir"]))
        sparse_model_dir = Path(context["sparse_model_dir"])
        work_dir = Path(context["work_dir"])

        dense_dir = work_dir / "dense"
        if dense_dir.exists():
            shutil.rmtree(dense_dir)
        dense_dir.mkdir(parents=True, exist_ok=True)

        if self.quality == "low":
            max_image_size = "640"
        elif self.quality == "high":
            max_image_size = "1000"
        else:
            max_image_size = "800"

        gpu_index = "-1" if context.get("no_gpu", False) else "0"

        self._run([
            "colmap", "image_undistorter",
            "--image_path", str(input_dir),
            "--input_path", str(sparse_model_dir),
            "--output_path", str(dense_dir),
            "--output_type", "COLMAP",
            "--max_image_size", max_image_size,
        ], "Undistort")

        self._run([
            "colmap", "patch_match_stereo",
            "--workspace_path", str(dense_dir),
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.gpu_index", gpu_index,
            "--PatchMatchStereo.geom_consistency", "false",
            "--PatchMatchStereo.num_samples", "10",
            "--PatchMatchStereo.num_iterations", "3",
            "--PatchMatchStereo.window_radius", "3",
            "--PatchMatchStereo.filter_min_ncc", "0.1",
        ], "Stereo")

        fused = dense_dir / "fused.ply"
        self._run([
            "colmap", "stereo_fusion",
            "--workspace_path", str(dense_dir),
            "--workspace_format", "COLMAP",
            "--input_type", "photometric",
            "--StereoFusion.min_num_pixels", "5",
            "--output_path", str(fused),
        ], "Fusion")

        if not fused.exists():
            raise RuntimeError("No fused point cloud")

        context["point_cloud"] = str(fused)
        context["dense_dir"] = str(dense_dir)
        return context
