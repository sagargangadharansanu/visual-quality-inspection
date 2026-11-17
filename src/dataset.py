"""
dataset.py - loads MVTec AD images for one category (e.g. metal_nut).

Folder layout this expects:
  data/mvtec/metal_nut/train/good/*.png          only good parts
  data/mvtec/metal_nut/test/<type>/*.png         good + defect types (bent, color, ...)
  data/mvtec/metal_nut/ground_truth/<type>/*_mask.png   where each defect is
"""
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# The pretrained network was trained on ImageNet photos with these colour
# statistics, so we scale our images the same way.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class MVTecDataset(Dataset):
    def __init__(self, root, category, split="train", image_size=224):
        self.root = Path(root) / category
        self.split = split

        # Image: resize, turn into numbers (tensor), normalise colours
        self.image_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        # Mask: resize with NEAREST so it stays pure black/white (no grey edges)
        self.mask_tf = transforms.Compose([
            transforms.Resize((image_size, image_size),
                              interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

        # Build a list of every image: (path, label, mask path, defect type)
        # label 0 = good (OK), 1 = defective (NOK)
        self.samples = []
        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Folder not found: {split_dir}")

        for type_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            defect_type = type_dir.name
            for img_path in sorted(type_dir.glob("*.png")):
                if defect_type == "good":
                    self.samples.append((img_path, 0, None, defect_type))
                else:
                    mask_path = (self.root / "ground_truth" / defect_type
                                 / f"{img_path.stem}_mask.png")
                    self.samples.append((img_path, 1, mask_path, defect_type))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, mask_path, defect_type = self.samples[idx]
        image = self.image_tf(Image.open(img_path).convert("RGB"))

        if mask_path is None:
            # Good part: empty mask (no defect anywhere)
            mask = torch.zeros(1, image.shape[1], image.shape[2])
        else:
            mask = self.mask_tf(Image.open(mask_path).convert("L"))
            mask = (mask > 0.5).float()

        return {
            "image": image,          # 3 x 224 x 224
            "label": label,          # 0 = OK, 1 = NOK
            "mask": mask,            # 1 x 224 x 224, 1 where the defect is
            "defect_type": defect_type,
            "path": str(img_path),
        }


# Run this file directly to check that the data loads correctly
if __name__ == "__main__":
    for split in ["train", "test"]:
        ds = MVTecDataset("data/mvtec", "metal_nut", split=split)
        counts = Counter(s[3] for s in ds.samples)
        print(f"{split}: {len(ds)} images -> {dict(counts)}")

    sample = MVTecDataset("data/mvtec", "metal_nut", split="test")[-1]
    print("image shape:", tuple(sample["image"].shape),
          "| mask shape:", tuple(sample["mask"].shape),
          "| label:", sample["label"],
          "| defect pixels:", int(sample["mask"].sum()))