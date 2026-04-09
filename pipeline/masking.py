import sys
import shutil
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from pipeline.base import Stage


class MaskingStage(Stage):
    """Generate foreground masks using MODNet for head/person segmentation."""

    def __init__(self, weights_path: str = "./modnet.ckpt", ref_size: int = 512):
        super().__init__("Masking")
        self.weights_path = weights_path
        self.ref_size = ref_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_modnet(self):
        if not Path(self.weights_path).exists():
            raise FileNotFoundError(f"MODNet weights not found: {self.weights_path}")

        modnet_src = Path("./modnet_src").resolve()
        src_path = str(modnet_src / "src")
        models_path = str(modnet_src / "src" / "models")

        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        if models_path not in sys.path:
            sys.path.insert(0, models_path)

        from models.modnet import MODNet

        modnet = MODNet(backbone_pretrained=False)
        weights = torch.load(self.weights_path, map_location=self.device)
        if isinstance(weights, dict) and "state_dict" in weights:
            weights = weights["state_dict"]

        modnet.load_state_dict(weights, strict=False)
        modnet.to(self.device)
        modnet.eval()

        print(f"[Masking] MODNet loaded on {self.device}")
        return modnet

    def _infer_mask(self, modnet, img_path: Path) -> np.ndarray:
        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        if max(H, W) > self.ref_size:
            if H > W:
                new_H, new_W = self.ref_size, int(W * self.ref_size / H)
            else:
                new_H, new_W = int(H * self.ref_size / W), self.ref_size
        else:
            new_H, new_W = H, W

        new_H = max(32, new_H - (new_H % 32))
        new_W = max(32, new_W - (new_W % 32))

        transform = transforms.Compose([
            transforms.Resize((new_H, new_W)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        inp = transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, _, matte = modnet(inp, True)

        matte = matte.squeeze().cpu().numpy()
        matte = Image.fromarray((matte * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
        return np.array(matte)

    def run(self, context: dict) -> dict:
        input_dir = Path(context["input_dir"])
        work_dir = Path(context["work_dir"])

        mask_dir = work_dir / "masks"
        masked_dir = work_dir / "images_masked"

        if mask_dir.exists():
            shutil.rmtree(mask_dir)
        if masked_dir.exists():
            shutil.rmtree(masked_dir)

        mask_dir.mkdir(parents=True, exist_ok=True)
        masked_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(
            list(input_dir.rglob("*.jpg")) +
            list(input_dir.rglob("*.jpeg")) +
            list(input_dir.rglob("*.png"))
        )
        print(f"[Masking] Processing {len(images)} images on {self.device}")

        modnet = self._load_modnet()

        for img_path in images:
            rel = img_path.relative_to(input_dir)

            mask_path = mask_dir / rel.parent / f"{rel.name}.png"
            masked_img_path = masked_dir / rel

            mask_path.parent.mkdir(parents=True, exist_ok=True)
            masked_img_path.parent.mkdir(parents=True, exist_ok=True)

            matte = self._infer_mask(modnet, img_path)

            mask_bin = (matte > 127).astype(np.uint8) * 255
            Image.fromarray(mask_bin).save(mask_path)

            img = np.array(Image.open(img_path).convert("RGB")).astype(np.uint8)
            masked = img.copy()
            masked[mask_bin == 0] = 0
            Image.fromarray(masked).save(masked_img_path)

        context["mask_dir"] = str(mask_dir)
        context["sfm_input_dir"] = str(input_dir)
        context["dense_input_dir"] = str(masked_dir)
        return context
