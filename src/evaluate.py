r"""
evaluate.py - full evaluation of PatchCore on one MVTec AD category.

Usage:
  python src\evaluate.py                     # metal_nut, reuses saved memory bank
  python src\evaluate.py --category screw    # another category
  python src\evaluate.py --refit             # force rebuilding the memory bank

Outputs (in results/<category>_YYYYMMDD_HHMM/):
  metrics.json          image AUROC, pixel AUROC, threshold analysis
  examples.png          original | true defect | heatmap, one per type
  score_histogram.png   good vs defective scores + threshold
"""
import argparse
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                 # save figures without opening windows
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import IMAGENET_MEAN, IMAGENET_STD, MVTecDataset
from features import FeatureExtractor
from patchcore import PatchCore

DATA_ROOT = "data/mvtec"
STAMP = datetime.now().strftime("%Y%m%d_%H%M")


def to_rgb(img):
    """Undo the ImageNet normalisation so the image can be shown."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def get_model(category, refit):
    """Load the newest saved memory bank, or build and save a new one."""
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model = PatchCore(FeatureExtractor(), coreset_ratio=0.1)

    saved = sorted(model_dir.glob(f"memory_bank_{category}_*.pt"))
    if saved and not refit:
        print(f"Loading memory bank: {saved[-1]}")
        model.memory_bank = torch.load(saved[-1])
    else:
        train_ds = MVTecDataset(DATA_ROOT, category, split="train")
        model.fit(DataLoader(train_ds, batch_size=8))
        path = model_dir / f"memory_bank_{category}_{STAMP}.pt"
        torch.save(model.memory_bank, path)
        print(f"Saved memory bank: {path}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="metal_nut")
    parser.add_argument("--refit", action="store_true")
    args = parser.parse_args()

    start = time.time()
    out_dir = Path("results") / f"{args.category}_{STAMP}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = get_model(args.category, args.refit)
    test_ds = MVTecDataset(DATA_ROOT, args.category, split="test")

    # ---- Score every test image ----
    scores, maps, masks, labels, types = [], [], [], [], []
    for batch in tqdm(DataLoader(test_ds, batch_size=8), desc="scoring (test)"):
        s, m = model.predict(batch["image"])
        scores.append(s.numpy())
        maps.append(m.squeeze(1).numpy())
        masks.append(batch["mask"].squeeze(1).numpy())
        labels += batch["label"].tolist()
        types += batch["defect_type"]
    scores = np.concatenate(scores)
    maps = np.concatenate(maps)
    masks = np.concatenate(masks).astype(int)
    labels = np.array(labels)
    types = np.array(types)

    # ---- 1. AUROC ----
    image_auroc = roc_auc_score(labels, scores)
    pixel_auroc = roc_auc_score(masks.ravel(), maps.ravel())

    # ---- 2. Threshold analysis (quality-engineer view) ----
    good, bad = scores[labels == 0], scores[labels == 1]

    # a) Zero escapes: reject everything at least as bad as the mildest defect
    thr_catch_all = float(bad.min())
    false_rejects = int((good >= thr_catch_all).sum())

    # b) Zero false rejects: only reject parts worse than the worst good part
    thr_no_false = float(good.max())
    escapes = int((bad <= thr_no_false).sum())

    mean_by_type = defaultdict(list)
    for t, s in zip(types, scores):
        mean_by_type[t].append(float(s))
    mean_by_type = {t: round(float(np.mean(v)), 3) for t, v in sorted(mean_by_type.items())}

    metrics = {
        "category": args.category,
        "n_test_good": int(len(good)),
        "n_test_defective": int(len(bad)),
        "image_auroc": round(float(image_auroc), 4),
        "pixel_auroc": round(float(pixel_auroc), 4),
        "zero_escapes": {
            "threshold": round(thr_catch_all, 4),
            "false_rejects": false_rejects,
            "false_reject_rate": round(false_rejects / len(good), 4),
        },
        "zero_false_rejects": {
            "threshold": round(thr_no_false, 4),
            "escapes": escapes,
            "escape_rate": round(escapes / len(bad), 4),
        },
        "mean_score_by_type": mean_by_type,
        "memory_bank_size": int(len(model.memory_bank)),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- 3. Figure: score histogram ----
    plt.figure(figsize=(7, 4))
    bins = np.linspace(scores.min(), scores.max(), 30)
    plt.hist(good, bins=bins, alpha=0.6, label=f"good (n={len(good)})", color="tab:green")
    plt.hist(bad, bins=bins, alpha=0.6, label=f"defective (n={len(bad)})", color="tab:red")
    plt.axvline(thr_catch_all, color="black", linestyle="--",
                label=f"zero-escape threshold ({false_rejects} false rejects)")
    plt.xlabel("anomaly score (higher = more unusual)")
    plt.ylabel("number of parts")
    plt.title(f"{args.category}: anomaly scores")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "score_histogram.png", dpi=150)
    plt.close()

    # ---- 4. Figure: one example per type ----
    type_names = sorted(set(types))
    vmin, vmax = np.percentile(maps, 1), maps.max()   # same colour scale for all
    fig, axes = plt.subplots(3, len(type_names), figsize=(3 * len(type_names), 9))
    for col, t in enumerate(type_names):
        i = int(np.where(types == t)[0][0])          # first image of this type
        sample = test_ds[i]
        rgb = to_rgb(sample["image"])
        verdict = "NOK" if scores[i] >= thr_catch_all else "OK"

        axes[0, col].imshow(rgb)
        axes[0, col].set_title(t)
        axes[1, col].imshow(masks[i], cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title("true defect")
        axes[2, col].imshow(rgb)
        axes[2, col].imshow(maps[i], cmap="jet", alpha=0.5, vmin=vmin, vmax=vmax)
        axes[2, col].set_title(f"score {scores[i]:.2f} -> {verdict}")
        for r in range(3):
            axes[r, col].axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "examples.png", dpi=120)
    plt.close()

    # ---- Report ----
    print(f"\n=== {args.category} ===")
    print(f"Image AUROC: {image_auroc:.3f}   (OK/NOK decision)")
    print(f"Pixel AUROC: {pixel_auroc:.3f}   (heatmap points at the right spot)")
    print(f"Zero escapes:       threshold {thr_catch_all:.3f} -> "
          f"{false_rejects}/{len(good)} good parts falsely rejected")
    print(f"Zero false rejects: threshold {thr_no_false:.3f} -> "
          f"{escapes}/{len(bad)} defects would escape")
    print(f"Saved to: {out_dir}")
    print(f"Total time: {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()