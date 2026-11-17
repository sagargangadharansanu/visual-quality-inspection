"""
patchcore.py - PatchCore anomaly detection (Roth et al., 2022), written in plain PyTorch.

fit():     build a memory bank of patch features from GOOD images only
predict(): for each test patch, distance to its nearest good patch
           -> anomaly heatmap per image + one anomaly score per image
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm


def greedy_coreset(features, ratio=0.1, proj_dim=128, seed=0):
    """Pick a small, well spread-out subset of patches.
    Each new pick = the patch furthest from everything already picked."""
    N, D = features.shape
    n_select = max(1, int(N * ratio))
    g = torch.Generator().manual_seed(seed)

    # Random projection: compare 128 numbers instead of 512 (much faster,
    # distances stay roughly the same)
    proj = torch.randn(D, proj_dim, generator=g) / proj_dim ** 0.5
    reduced = features @ proj

    first = int(torch.randint(N, (1,), generator=g))
    selected = [first]
    min_dist = torch.linalg.norm(reduced - reduced[first], dim=1)

    for _ in tqdm(range(n_select - 1), desc="coreset"):
        idx = int(torch.argmax(min_dist))      # furthest from current set
        selected.append(idx)
        new_dist = torch.linalg.norm(reduced - reduced[idx], dim=1)
        min_dist = torch.minimum(min_dist, new_dist)

    return torch.tensor(selected)


def gaussian_blur(x, sigma=4):
    """Smooth heatmaps. x: B x 1 x H x W"""
    k = int(4 * sigma) + 1                     # kernel size, e.g. 17
    ax = torch.arange(k, dtype=torch.float32) - k // 2
    kernel = torch.exp(-ax ** 2 / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    x = F.pad(x, (k // 2,) * 4, mode="reflect")
    x = F.conv2d(x, kernel.view(1, 1, 1, k))   # blur horizontally
    x = F.conv2d(x, kernel.view(1, 1, k, 1))   # blur vertically
    return x


class PatchCore:
    def __init__(self, extractor, coreset_ratio=0.1, image_size=224):
        self.extractor = extractor
        self.coreset_ratio = coreset_ratio
        self.image_size = image_size
        self.memory_bank = None

    def fit(self, train_loader):
        all_feats = []
        for batch in tqdm(train_loader, desc="features (train)"):
            f = self.extractor(batch["image"])            # B x 28 x 28 x D
            all_feats.append(f.reshape(-1, f.shape[-1]))  # (B*784) x D
        all_feats = torch.cat(all_feats)
        print(f"patches from good images: {len(all_feats):,}")

        idx = greedy_coreset(all_feats, ratio=self.coreset_ratio)
        self.memory_bank = all_feats[idx].contiguous()
        print(f"memory bank after coreset: {len(self.memory_bank):,}")

    @torch.no_grad()
    def predict(self, images):
        """images: B x 3 x 224 x 224
        returns scores (B,) and heatmaps (B x 1 x 224 x 224)"""
        feats = self.extractor(images)                    # B x 28 x 28 x D
        B, H, W, D = feats.shape
        maps = []
        for i in range(B):
            patches = feats[i].reshape(-1, D)                     # 784 x D
            dist = torch.cdist(patches, self.memory_bank)         # 784 x bank
            nearest = dist.min(dim=1).values                      # 784
            maps.append(nearest.reshape(1, H, W))
        maps = torch.stack(maps)                                  # B x 1 x 28 x 28

        maps = F.interpolate(maps, size=(self.image_size, self.image_size),
                             mode="bilinear", align_corners=False)
        maps = gaussian_blur(maps, sigma=4)
        scores = maps.amax(dim=(1, 2, 3))         # worst spot = part score
        return scores, maps


# Self-test: fit on metal_nut, score the test set, report image-level AUROC
if __name__ == "__main__":
    import time
    from collections import defaultdict
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader
    from dataset import MVTecDataset
    from features import FeatureExtractor

    start = time.time()
    train_ds = MVTecDataset("data/mvtec", "metal_nut", split="train")
    test_ds = MVTecDataset("data/mvtec", "metal_nut", split="test")

    model = PatchCore(FeatureExtractor(), coreset_ratio=0.1)
    model.fit(DataLoader(train_ds, batch_size=8))

    labels, scores, types = [], [], []
    for batch in tqdm(DataLoader(test_ds, batch_size=8), desc="scoring (test)"):
        s, _ = model.predict(batch["image"])
        scores += s.tolist()
        labels += batch["label"].tolist()
        types += batch["defect_type"]

    print(f"\nImage-level AUROC: {roc_auc_score(labels, scores):.3f}")

    by_type = defaultdict(list)
    for t, s in zip(types, scores):
        by_type[t].append(s)
    print("Mean anomaly score per type:")
    for t, vals in sorted(by_type.items()):
        print(f"  {t:8s} {sum(vals)/len(vals):.2f}  (n={len(vals)})")
    print(f"Total time: {(time.time()-start)/60:.1f} min")