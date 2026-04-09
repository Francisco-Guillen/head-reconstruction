import shutil
from pathlib import Path
from PIL import Image
from pipeline.base import Stage


class PreprocessingStage(Stage):
    """Convert input images to RGB and prepare for COLMAP."""

    def __init__(self):
        super().__init__("Preprocessing")

    def run(self, context: dict) -> dict:
        input_dir = Path(context["input_dir"])
        work_dir = Path(context["work_dir"])

        out_dir = work_dir / "images_rgb"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        supported = {".png", ".jpg", ".jpeg"}
        images = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in supported])
        print(f"[Preprocessing] Found {len(images)} images")

        for img_path in images:
            img = Image.open(img_path)
            if img.mode == "RGBA":
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            else:
                img = img.convert("RGB")
            img.save(out_dir / f"{img_path.stem}.jpg", quality=95)

        print(f"[Preprocessing] Saved {len(images)} images to {out_dir}")
        context["input_dir"] = str(out_dir)
        return context
