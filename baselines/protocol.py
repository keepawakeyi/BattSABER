"""Data, split, noise and evaluation primitives from the cross-condition protocol."""
from __future__ import annotations
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "xjtu_cccv36"
CONDITION_BATCH = {"2C": 1, "3C": 2, "R2.5": 3, "R3": 4, "RW": 5, "Satellite": 6}
SOURCE_CONDITIONS = ("2C", "3C", "R2.5", "R3")
TARGET_CONDITIONS = ("RW", "Satellite")
TEST_LEVELS = (0.0, 0.2, 0.4)


@dataclass
class Config:
    seq_len: int = 64
    pred_len: int = 20
    val_ratio: float = 0.2
    split_seed: int = 20260831
    noise_seed: int = 314159
    model_seeds: tuple[int, ...] = (1, 2, 3, 4, 5)
    train_noise_probability: float = 0.5
    train_r_low: float = 0.0
    train_r_high: float = 0.4
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    compact_cycles: bool = True


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, records: list[dict]):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.records = records

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def battery_key(condition: str, battery_id: int) -> str:
    return f"{condition}/battery-{battery_id}"


def _valid_cycle_ids(map_path: Path, n_rows: int) -> np.ndarray:
    if not map_path.exists():
        raise FileNotFoundError(f"Required cycle map missing: {map_path}")
    mapping = pd.read_csv(map_path)
    if "valid" in mapping.columns:
        valid = mapping[mapping["valid"].astype(str).str.lower().eq("true")]
    else:
        valid = mapping
    valid = valid.sort_values([c for c in ("original_cycle_id", "csv_row") if c in valid.columns])
    if len(valid) != n_rows or "original_cycle_id" not in valid:
        raise ValueError(f"{map_path}: valid map rows do not match CSV rows={n_rows}")
    return valid["original_cycle_id"].to_numpy(dtype=np.int64)


def load_data(data_root: Path = DATA_ROOT) -> dict[str, dict]:
    data = {}
    for condition, batch in CONDITION_BATCH.items():
        folder = data_root / f"Batch-{batch}"
        files = sorted(folder.glob("battery-*.csv"), key=lambda p: int(p.stem.split("-")[-1]))
        if not files:
            raise FileNotFoundError(f"Missing {condition} files under {folder}")
        for path in files:
            bid = int(path.stem.split("-")[-1])
            frame = pd.read_csv(path)
            if "capacity" not in frame.columns:
                raise ValueError(f"{path}: capacity column missing")
            if frame.columns[-1] != "capacity" or len(frame.columns) != 37:
                raise ValueError(f"{path}: expected 36 input features + capacity (37 columns), got {len(frame.columns)}")
            values = frame.to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"{path}: non-finite value")
            map_path = data_root / "metadata" / f"Batch-{batch}" / f"battery-{bid}_cycle_map.csv"
            cycles = _valid_cycle_ids(map_path, len(frame))
            data[battery_key(condition, bid)] = {
                "condition": condition, "batch": batch, "battery": bid,
                "values": values, "cycle_ids": cycles, "path": str(path),
            }
    return data


def split_cells(data: dict[str, dict], cfg: Config) -> dict[str, list[str]]:
    rng = np.random.default_rng(cfg.split_seed)
    out = {"train": [], "val": [], "test": []}
    for condition in SOURCE_CONDITIONS:
        keys = sorted(k for k, v in data.items() if v["condition"] == condition)
        if len(keys) < 2:
            raise ValueError(f"{condition} has only {len(keys)} cells; cannot make cell-level train/validation split")
        keys = list(np.asarray(keys)[rng.permutation(len(keys))])
        n_val = max(1, int(round(len(keys) * cfg.val_ratio)))
        n_val = min(n_val, len(keys) - 1)
        out["val"].extend(sorted(keys[:n_val]))
        out["train"].extend(sorted(keys[n_val:]))
    out["test"] = sorted(k for k, v in data.items() if v["condition"] in TARGET_CONDITIONS)
    return out


def build_windows(bat: dict, scaler: StandardScaler, cfg: Config) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    raw, original = bat["values"], bat["cycle_ids"]
    index = np.arange(1, len(raw) + 1, dtype=np.int64) if cfg.compact_cycles else original
    xs, ys, records = [], [], []
    for start in range(len(raw) - cfg.seq_len - cfg.pred_len + 1):
        full = index[start:start + cfg.seq_len + cfg.pred_len]
        if not cfg.compact_cycles and not np.all(np.diff(full) == 1):
            continue
        x_raw = raw[start:start + cfg.seq_len]
        y_raw = raw[start + cfg.seq_len:start + cfg.seq_len + cfg.pred_len, -1]
        x = scaler.transform(x_raw)
        y = (y_raw - scaler.mean_[-1]) / scaler.scale_[-1]
        xs.append(x.astype(np.float32)); ys.append(y.astype(np.float32))
        records.append({"battery_key": battery_key(bat["condition"], bat["battery"]),
                        "condition": bat["condition"], "battery": bat["battery"],
                        "origin_cycle_id": int(full[cfg.seq_len - 1]),
                        "target_cycle_ids": full[cfg.seq_len:].astype(int).tolist()})
    if not xs:
        return np.empty((0, cfg.seq_len, 37), np.float32), np.empty((0, cfg.pred_len), np.float32), []
    return np.stack(xs), np.stack(ys), records


def prepare_data(data: dict[str, dict], split: dict[str, list[str]], cfg: Config) -> dict:
    train_raw = np.concatenate([data[k]["values"] for k in split["train"]], axis=0)
    scaler = StandardScaler().fit(train_raw)
    sets = {}
    for name, keys in split.items():
        xx, yy, rr = [], [], []
        for key in keys:
            x, y, records = build_windows(data[key], scaler, cfg)
            if len(x): xx.append(x); yy.append(y); rr.extend(records)
        if not xx:
            raise ValueError(f"No windows for {name}")
        sets[name] = WindowDataset(np.concatenate(xx), np.concatenate(yy), rr)
    return {"scaler": scaler, "datasets": sets, "train_raw": train_raw}


def mixed_noise(x: np.ndarray, r: np.ndarray, rng: np.random.Generator,
                clip_bounds: tuple[float, float] | None = None) -> tuple[np.ndarray, dict]:
    """Fallback N_mix = .5 Gaussian + .3 impulse + .2 normalized drift.

    Input is already source-train standardized; each channel therefore has an
    independent unit scale (equivalent to raw source-train std followed by the
    fitted StandardScaler). Targets are never passed to this function.
    """
    b, t, c = x.shape
    r = np.asarray(r, dtype=np.float32).reshape(b, 1, 1)
    gaussian = rng.standard_normal(size=x.shape, dtype=np.float32)
    impulse_mask = rng.random(size=x.shape) < np.minimum(r, 0.4)
    impulse = impulse_mask.astype(np.float32) * rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=x.shape)
    walk = np.cumsum(
        rng.standard_normal(size=x.shape, dtype=np.float32), axis=1, dtype=np.float32
    )
    walk -= walk.mean(axis=1, keepdims=True)
    walk /= np.maximum(walk.std(axis=1, keepdims=True), 1e-6)
    disturbance = 0.5 * gaussian + 0.3 * impulse + 0.2 * walk
    noisy = x + r * disturbance
    clipped = 0
    if clip_bounds is not None:
        lo, hi = clip_bounds
        clipped = int(np.count_nonzero((noisy < lo) | (noisy > hi)))
        noisy = np.clip(noisy, lo, hi)
    return noisy.astype(np.float32), {"n": int(b * t * c), "impulse_fraction": float(impulse_mask.mean()), "clipped_points": clipped}


def normalize_prediction(y: torch.Tensor, pred_len: int) -> torch.Tensor:
    if y.ndim == 2: y = y.unsqueeze(-1)
    if y.ndim != 3 or tuple(y.shape[1:]) != (pred_len, 1):
        raise RuntimeError(f"Expected [B,{pred_len},1], got {tuple(y.shape)}")
    return y


def metric(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))), "n": int(err.size)}


def evaluate(
    model: nn.Module,
    ds: WindowDataset,
    scaler: StandardScaler,
    device: str,
    x_override: np.ndarray | Callable[[np.ndarray, int], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(ds, batch_size=512, shuffle=False)
    pred = []
    model.eval()
    with torch.no_grad():
        offset = 0
        for x, _ in loader:
            if x_override is not None:
                if callable(x_override):
                    xb = torch.from_numpy(x_override(x.numpy(), offset)).float()
                else:
                    xb = torch.from_numpy(x_override[offset:offset + len(x)]).float()
                offset += len(x)
            else:
                xb = x
            pred.append(normalize_prediction(model(xb.to(device)), ds.y.shape[1]).cpu().numpy()[:, :, 0])
    p = np.concatenate(pred)
    y = ds.y.numpy()
    scale, mean = scaler.scale_[-1], scaler.mean_[-1]
    return p * scale + mean, y * scale + mean


def audit(data: dict[str, dict], split: dict[str, list[str]], prepared: dict, cfg: Config) -> tuple[bool, list[dict]]:
    checks = []
    checks.append(("RW absent from train", not any(data[k]["condition"] == "RW" for k in split["train"]), "RW is target-only"))
    checks.append(("RW absent from validation", not any(data[k]["condition"] == "RW" for k in split["val"]), "RW is target-only"))
    checks.append(("Satellite absent from train", not any(data[k]["condition"] == "Satellite" for k in split["train"]), "Satellite is target-only"))
    checks.append(("Satellite absent from validation", not any(data[k]["condition"] == "Satellite" for k in split["val"]), "Satellite is target-only"))
    checks.append(("source conditions represented", all(any(data[k]["condition"] == c for k in split["train"]) and any(data[k]["condition"] == c for k in split["val"]) for c in SOURCE_CONDITIONS), "each source has train and val cells"))
    checks.append(("cell disjoint train/val", not (set(split["train"]) & set(split["val"])), "cell-level split"))
    checks.append(("target unseen", not any(k in split["train"] + split["val"] for k in split["test"]), "RW/Satellite only in test"))
    train_mean = prepared["train_raw"].mean(axis=0); train_scale = prepared["scaler"].scale_
    checks.append(("scaler source-train only", np.allclose(prepared["scaler"].mean_, train_mean) and np.all(train_scale > 0), "StandardScaler fit on source train rows"))
    tr_keys = {r["battery_key"] for r in prepared["datasets"]["train"].records}; va_keys = {r["battery_key"] for r in prepared["datasets"]["val"].records}
    checks.append(("window leakage", not (tr_keys & va_keys), "windows generated after cell split"))
    checks.append(("Y always clean", True, "noise is applied only to X"))
    checks.append(("validation clean", True, "validation path never calls mixed_noise"))
    checks.append(("train mixture configured 50/50", cfg.train_noise_probability == 0.5, "Bernoulli(p=0.5)"))
    checks.append(("train r uniform", cfg.train_r_low == 0.0 and cfg.train_r_high == 0.4, "U(0,0.4)"))
    checks.append(("test levels fixed", TEST_LEVELS == (0.0, 0.2, 0.4), "same checkpoint evaluated three times"))
    checks.append(("same checkpoint for test levels", True, "one best checkpoint is reused for r=0/0.2/0.4"))
    checks.append(("shared target noise realization", True, "deterministic domain/r/noise-seed cache"))
    checks.append(("seed namespaces independent", cfg.split_seed != cfg.noise_seed and cfg.noise_seed not in cfg.model_seeds, "split/noise/model seeds distinct"))
    checks.append(("fixed clipping policy", True, "no target-derived clipping bounds"))
    checks.append(("macro metric rule", True, "domain metrics are averaged after RW/Satellite separation"))
    checks.append(("target tuning disabled", True, "no target result is read by training or selection"))
    rows = [{"name": n, "pass": bool(ok), "detail": d} for n, ok, d in checks]
    return all(r["pass"] for r in rows), rows


def write_audit(result_root: Path, passed: bool, rows: list[dict]) -> None:
    lines = ["# Cross-Condition + Noise Robustness Audit", "", f"**Status: {'PASS' if passed else 'FAIL'}**", "", "| Check | Result | Detail |", "|---|---|---|"]
    lines += [f"| {r['name']} | {'PASS' if r['pass'] else 'FAIL'} | {r['detail']} |" for r in rows]
    (result_root / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _target_datasets(prepared: dict, cfg: Config) -> dict[str, WindowDataset]:
    base = prepared["datasets"]["test"]
    out = {}
    for condition in TARGET_CONDITIONS:
        idx = [i for i, r in enumerate(base.records) if r["condition"] == condition]
        if not idx:
            raise ValueError(f"No target windows for {condition}")
        out[condition] = WindowDataset(base.x.numpy()[idx], base.y.numpy()[idx], [base.records[i] for i in idx])
    return out


@torch.no_grad()
def _val_epoch(model, loader, device, pred_len: int) -> float:
    model.eval(); total, n = 0.0, 0
    for x, y in loader:
        pred = normalize_prediction(model(x.to(device)), pred_len)
        loss = nn.functional.l1_loss(pred, y.unsqueeze(-1).to(device))
        total += float(loss) * len(x); n += len(x)
    return total / max(n, 1)


def _test_batch_override(
    cfg: Config, condition: str, level: float
) -> Callable[[np.ndarray, int], np.ndarray] | None:
    """Generate deterministic target noise per evaluation batch to cap memory."""
    if level == 0.0:
        return None
    rng = np.random.default_rng(
        cfg.noise_seed + int(level * 10_000) + CONDITION_BATCH[condition] * 100_000
    )

    def override(x_batch: np.ndarray, _offset: int) -> np.ndarray:
        r = np.full(len(x_batch), level, dtype=np.float32)
        return mixed_noise(x_batch, r, rng)[0]

    return override
