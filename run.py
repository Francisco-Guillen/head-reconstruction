import argparse
from pathlib import Path

from pipeline.preprocessing import PreprocessingStage
from pipeline.masking import MaskingStage
from pipeline.sfm import SfMStage
from pipeline.dense import DenseStage
from pipeline.meshing import MeshingStage
from pipeline.postprocess import PostProcessStage
from utils.timing import print_summary


def parse_args():
    parser = argparse.ArgumentParser(description="3D Head Reconstruction Pipeline")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quality", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--work-dir", default="./workdir")
    parser.add_argument("--max-faces", type=int, default=50000)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--weights", default="./modnet.ckpt")
    return parser.parse_args()


def main():
    args = parse_args()

    context = {
        "input_dir":   str(Path(args.input).resolve()),
        "output_path": str(Path(args.output).resolve()),
        "work_dir":    str(Path(args.work_dir).resolve()),
        "quality":     args.quality,
        "max_faces":   args.max_faces,
        "no_gpu":      args.no_gpu,
    }

    input_dir = Path(context["input_dir"])
    output_path = Path(context["output_path"])

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if output_path.suffix.lower() not in [".obj", ".ply"]:
        raise ValueError("Output must be .obj or .ply")

    Path(context["work_dir"]).mkdir(parents=True, exist_ok=True)

    print("=== 3D Head Reconstruction Pipeline ===")
    print(f"  Input:     {context['input_dir']}")
    print(f"  Output:    {context['output_path']}")
    print(f"  Quality:   {context['quality']}")
    print(f"  Work dir:  {context['work_dir']}")
    print(f"  Max faces: {context['max_faces']}")
    print(f"  No GPU:    {context['no_gpu']}")
    print(f"  Weights:   {args.weights}")
    print("========================================\n")

    poisson_depth = 8 if args.quality == "low" else 9

    pipeline = [
        PreprocessingStage(),
        MaskingStage(weights_path=args.weights),
        SfMStage(quality=args.quality, use_masks=False),
        DenseStage(quality=args.quality),
        MeshingStage(depth=poisson_depth, crop_bottom_percentile=0.0, radial_percentile=0.0),
        PostProcessStage(max_faces=args.max_faces, fill_holes=True),
    ]

    try:
        for stage in pipeline:
            context = stage(context)
    finally:
        print_summary()

    if "final_mesh" not in context:
        raise RuntimeError("Pipeline did not produce final mesh.")

    print(f"\n✅ Mesh saved to: {context['final_mesh']}")
    print(f"   Faces:    {context['final_mesh_num_faces']}")
    print(f"   Vertices: {context['final_mesh_num_vertices']}")


if __name__ == "__main__":
    main()
