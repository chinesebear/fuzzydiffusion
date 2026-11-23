"""Representative (delegate) selection for the fuzzy rule antecedents.

The paper (Sec. III.B, Eq. 9) selects representatives with K-Medoids over
image features and stores the real samples as the prototypes of the fuzzy
sets used by every rule chain.

The feature used here is the same space as SMC: L2-normalised
``[BERT(BLIP(image)) ; VGG16(image)]``.  K-Medoids is implemented directly
(PAM-style assignment/update) so no extra clustering dependency is needed.
"""

from __future__ import annotations

import csv
import itertools
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def feature_matrix(images: torch.Tensor, membership) -> torch.Tensor:
    """Return the feature matrix used by K-Medoids.

    ``images`` is ``[N, C, H, W]`` in ``[-1, 1]``.  Both SMC branches are
    already L2-normalised, so they are concatenated and re-normalised.
    """
    references = membership.reference_features(images)
    features = torch.cat([references["sem"], references["vis"]], dim=-1)
    return F.normalize(features, dim=-1)


def _distance_matrix(features: torch.Tensor) -> np.ndarray:
    """Cosine distance matrix (features are already normalised)."""
    sim = (features @ features.T).detach().cpu().numpy()
    return np.clip(1.0 - sim, 0.0, None)


def kmedoids(features: torch.Tensor, k: int, max_iter: int = 100, seed: int = 0):
    """PAM-style K-Medoids on a cosine distance matrix.

    Returns the indices of the ``k`` selected real samples.
    """
    n = int(features.shape[0])
    if k >= n:
        return list(range(n))
    if k <= 0:
        raise ValueError("k must be positive")

    distances = _distance_matrix(features)
    rng = np.random.default_rng(seed)
    medoids = rng.choice(n, size=k, replace=False)

    for _ in range(max_iter):
        assignment = distances[:, medoids].argmin(axis=1)

        new_medoids = []
        for cluster in range(k):
            members = np.where(assignment == cluster)[0]
            if members.size == 0:
                new_medoids.append(int(medoids[cluster]))
                continue
            # The medoid minimises the sum of distances inside the cluster.
            intra = distances[np.ix_(members, members)]
            new_medoids.append(int(members[int(intra.sum(axis=1).argmin())]))

        new_medoids = list(dict.fromkeys(new_medoids))
        if len(new_medoids) < k:
            # Fill empty clusters with the points farthest from the current
            # medoid set (deterministic).
            nearest = distances[:, new_medoids].min(axis=1)
            for index in np.argsort(nearest)[::-1]:
                index = int(index)
                if index not in new_medoids:
                    new_medoids.append(index)
                if len(new_medoids) == k:
                    break

        new_medoids = np.array(sorted(new_medoids), dtype=np.int64)
        if np.array_equal(new_medoids, medoids):
            break
        medoids = new_medoids

    return [int(index) for index in medoids]


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = ((image.detach().float().clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(0.0, 1.0)
    return transforms.ToPILImage()(image.cpu())


def save_delegates(delegates: torch.Tensor, save_dir: str, name: str, k: int):
    """Save representative images and their CSV list (paper Eq. 9)."""
    os.makedirs(save_dir, exist_ok=True)
    paths = []
    for index, image in enumerate(delegates):
        path = os.path.join(save_dir, f"{name}_{k}_delegates_{index + 1}.jpg")
        _tensor_to_pil(image).save(path)
        paths.append(path)

    csv_path = os.path.join(save_dir, f"{name}_{k}_delegates.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["delegates"])
        for path in paths:
            writer.writerow([path])
    return csv_path, paths


def load_delegates(csv_path: str, transform=None):
    """Load representative images listed in a delegates CSV file."""
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    delegates = []
    with open(csv_path, encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header and header[0] != "delegates":
            handle.seek(0)  # tolerate header-less files
            reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            image = Image.open(row[0]).convert("RGB")
            delegates.append(transform(image))
            image.close()
    return delegates


def collect_images(loader, max_batches: int = 10):
    """Collect a fixed number of batches from a ``DataLoader``."""
    images = []
    for batch in itertools.islice(loader, max_batches):
        if isinstance(batch, dict):
            batch = batch["image"]
        images.append(batch.detach().cpu())
    if not images:
        raise RuntimeError("empty data loader, cannot select representatives")
    return torch.cat(images, dim=0)


def get_delegates(
    loader,
    membership,
    delegate_nums,
    save_dir: str,
    name: str = "dataset",
    max_batches: int = 10,
    seed: int = 0,
):
    """Select K-Medoids representatives for every requested number of rules.

    Returns ``{k: delegates_tensor}``.  The CSV files written by
    :func:`save_delegates` can be passed to the training scripts.
    """
    images = collect_images(loader, max_batches=max_batches)
    features = feature_matrix(images, membership)

    result = {}
    for index, k in enumerate(delegate_nums):
        medoid_indices = kmedoids(features, int(k), seed=seed + index)
        delegates = images[medoid_indices]
        result[int(k)] = delegates
        save_delegates(delegates, save_dir, name, int(k))
    return result
