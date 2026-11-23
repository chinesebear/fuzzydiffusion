"""Representative (delegate) selection for the fuzzy rule antecedents.

The paper (Sec. III.B, Eq. 9) uses K-Medoids clustering over image features
and keeps the real medoid samples as the prototypes of the K fuzzy sets.
This script replaces the earlier Fuzzy-C-Means / pixel-entropy prototype and
writes the CSV files consumed by :class:`fldm.FuzzyLatentDiffusion`.

Examples
--------
    python fcm.py --dataset lsun_church --rules 2 3 4 5
    python fcm.py --dataset coco --rules 3 --alpha 0.7
    python fcm.py --dataset cub200 --rules 3 --alpha 0.7
"""

from __future__ import annotations

import argparse
import os

from torch.utils.data import DataLoader
from torchvision import transforms

from loader import CELEBA_HQ, COCO, CUB200, FFHQ, LSUN
from membership import SimilarityMembership
from representatives import get_delegates
from setting import options

TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def build_dataset(name):
    if name == "lsun_church":
        return LSUN(options.base_path + "datasets/lsun", dataset_sub_path="churches",
                    phase="train", transform=TRANSFORM)
    if name == "lsun_bedroom":
        return LSUN(options.base_path + "datasets/lsun", dataset_sub_path="bedrooms",
                    phase="train", transform=TRANSFORM)
    if name == "coco":
        return COCO("ChristophSchuhmann/MS_COCO_2017_URL_TEXT",
                    phase="train", transform=TRANSFORM)
    if name == "cub200":
        return CUB200(phase="train", transform=TRANSFORM)
    if name == "ffhq":
        return FFHQ(options.base_path + "datasets/FFHQ", transform=TRANSFORM)
    if name == "celeba_hq":
        return CELEBA_HQ(options.base_path + "datasets/celeba_hq_256",
                         transform=TRANSFORM)
    raise ValueError(f"unknown dataset: {name}")


def select_delegates(dataset, name, delegate_nums, alpha, batch_size=32,
                     max_batches=10, seed=0):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
    )
    membership = SimilarityMembership(alpha=alpha, device=options.device)
    save_dir = options.base_path + f"output/delegates/{name}/"
    os.makedirs(save_dir, exist_ok=True)
    return get_delegates(
        loader,
        membership,
        delegate_nums,
        save_dir=save_dir,
        name=name,
        max_batches=max_batches,
        seed=seed,
    )


def main():
    parser = argparse.ArgumentParser(description="K-Medoids representative selection")
    parser.add_argument("--dataset", default="lsun_church",
                        choices=["lsun_church", "lsun_bedroom", "coco", "cub200",
                                 "ffhq", "celeba_hq"])
    parser.add_argument("--rules", type=int, nargs="+", default=[3],
                        help="number of rule chains / representatives")
    parser.add_argument("--alpha", type=float, default=None,
                        help="SMC balance (default: 0.3 LSUN, 0.7 others)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    alpha = args.alpha
    if alpha is None:
        alpha = 0.3 if args.dataset.startswith("lsun") else 0.7

    dataset = build_dataset(args.dataset)
    select_delegates(
        dataset,
        args.dataset,
        args.rules,
        alpha=alpha,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        seed=args.seed,
    )
    print("representatives saved to output/delegates/")


if __name__ == "__main__":
    main()
