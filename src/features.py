"""
features.py - turns images into patch features using a pretrained WideResNet-50.

Each 224x224 image becomes a 28x28 grid of patches, and each patch is
described by a vector of numbers (default 512). Similar-looking patches
get similar vectors - that's what lets us spot "never seen before" patches.
"""
import torch
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights


class FeatureExtractor(torch.nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        # Pretrained on ImageNet. Downloads ~130 MB the first time,
        # then cached in C:\Users\<you>\.cache\torch
        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        backbone.eval()                      # inference mode, no training
        for p in backbone.parameters():
            p.requires_grad = False          # freeze: we never change its weights
        self.backbone = backbone
        self.out_dim = out_dim

    @torch.no_grad()
    def forward(self, x):
        """x: batch of images, B x 3 x 224 x 224
        returns: B x 28 x 28 x out_dim  (one vector per patch)"""
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        x = b.layer1(x)
        f2 = b.layer2(x)          # B x  512 x 28 x 28  (finer detail)
        f3 = b.layer3(f2)         # B x 1024 x 14 x 14  (more context)

        # Each patch also looks at its 3x3 neighbours
        f2 = F.avg_pool2d(f2, kernel_size=3, stride=1, padding=1)
        f3 = F.avg_pool2d(f3, kernel_size=3, stride=1, padding=1)

        # Bring layer3 up to the 28x28 grid and stack both layers
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear",
                           align_corners=False)
        feats = torch.cat([f2, f3], dim=1)          # B x 1536 x 28 x 28

        # Shrink 1536 -> out_dim numbers per patch (saves memory)
        B, C, H, W = feats.shape
        feats = feats.permute(0, 2, 3, 1).reshape(-1, 1, C)
        feats = F.adaptive_avg_pool1d(feats, self.out_dim).squeeze(1)
        return feats.reshape(B, H, W, self.out_dim)


# Self-test: run features on one batch and estimate time + memory
if __name__ == "__main__":
    import time
    from torch.utils.data import DataLoader
    from dataset import MVTecDataset

    ds = MVTecDataset("data/mvtec", "metal_nut", split="train")
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    model = FeatureExtractor()

    batch = next(iter(loader))
    start = time.time()
    feats = model(batch["image"])
    secs = (time.time() - start) / len(batch["image"])

    n_patches = len(ds) * feats.shape[1] * feats.shape[2]
    mem_mb = n_patches * feats.shape[3] * 4 / 1e6
    print("feature shape:", tuple(feats.shape))
    print(f"time per image: {secs:.2f} s -> all {len(ds)} train images: ~{secs*len(ds)/60:.1f} min")
    print(f"total patches: {n_patches:,} -> ~{mem_mb:.0f} MB in memory")