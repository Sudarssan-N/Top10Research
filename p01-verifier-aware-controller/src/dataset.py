"""PyTorch datasets for cached controller features."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split


class FeatureRecord:
    def __init__(self, row: Dict):
        self.id = row["id"]
        self.benchmark = row.get("benchmark", "")
        self.hidden = row["hidden"]
        self.target_value = float(row["target_value"])
        self.target_trust = float(row["target_trust"])
        self.meta = row.get("labels_raw", {})


class CachedFeatureDataset(Dataset):
    def __init__(self, records: List[Dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r = self.records[idx]
        hidden = r["hidden"]
        if not isinstance(hidden, torch.Tensor):
            hidden = torch.tensor(hidden, dtype=torch.float32)
        return {
            "hidden": hidden.float(),
            "target": torch.tensor(r["target_value"], dtype=torch.float32),
            "target_trust": torch.tensor(r["target_trust"], dtype=torch.float32),
            "id": r["id"],
        }


def load_feature_cache(path: str) -> Tuple[Dict, List[Dict]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature cache not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload.get("meta", {}), payload.get("records", [])


def merge_feature_caches(paths: List[str]) -> Tuple[Dict, List[Dict]]:
    metas = []
    records = []
    for p in paths:
        meta, recs = load_feature_cache(p)
        metas.append(meta)
        records.extend(recs)
    if not metas:
        return {}, []
    merged_meta = dict(metas[0])
    merged_meta["sources"] = [m.get("benchmark", p) for m, p in zip(metas, paths)]
    merged_meta["num_records"] = len(records)
    return merged_meta, records


def build_dataloaders(
    records: List[Dict],
    batch_size: int = 8,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    dataset = CachedFeatureDataset(records)
    n = len(dataset)
    if n < 2:
        loader = DataLoader(dataset, batch_size=min(batch_size, n), shuffle=True)
        return loader, loader

    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader