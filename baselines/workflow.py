"""Audited training, source-only selection, and unseen-domain evaluation."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import protocol as p
from .factory import make_model

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("MLP", "CNN", "DLinear", "LDNet", "OmniTIEFormer", "ModernTCN", "GRU")
PAPER_METHODS = METHODS[:-1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare(data_root, cfg):
    data = p.load_data(Path(data_root))
    split = p.split_cells(data, cfg)
    expected = read_json(ROOT / "configs/split.json")
    if any(split[key] != expected[key] for key in ("train", "val", "test")):
        raise ValueError("Dataset does not reproduce the published source/target split")
    prepared = p.prepare_data(data, split, cfg)
    return data, split, prepared


def audit(data, split, prepared, cfg, selected, methods, output):
    _, checks = p.audit(data, split, prepared, cfg)
    counts = {c: sum(v["condition"] == c for v in data.values()) for c in p.CONDITION_BATCH}
    checks.append({"name": "55 batteries / six conditions", "pass": counts == {
        "2C": 8, "3C": 15, "R2.5": 8, "R3": 8, "RW": 8, "Satellite": 8}, "detail": counts})
    expected_columns = pd.read_csv(ROOT / "data/feature_manifest.csv")["feature_name"].tolist() + ["capacity"]
    schema_ok = all(pd.read_csv(v["path"], nrows=0).columns.tolist() == expected_columns for v in data.values())
    checks.append({"name": "feature order matches manifest", "pass": schema_ok, "detail": "36 features + capacity last"})
    for method in methods:
        model = make_model(method, selected[method]).eval()
        with torch.no_grad():
            y = p.normalize_prediction(model(torch.zeros(2, cfg.seq_len, 37)), cfg.pred_len)
        checks.append({"name": method + " finite forward", "pass": bool(torch.isfinite(y).all()), "detail": list(y.shape)})
    passed = all(row["pass"] for row in checks)
    output.mkdir(parents=True, exist_ok=True)
    p.write_audit(output, passed, checks)
    write_json(output / "split.json", split)
    write_json(output / "protocol.json", {
        "config": asdict(cfg), "battery_counts": counts, "methods": list(methods),
        "selected": {m: selected[m] for m in methods},
        "train_windows": len(prepared["datasets"]["train"]),
        "val_windows": len(prepared["datasets"]["val"]),
        "test_windows": len(prepared["datasets"]["test"]),
        "noise_formula": "r*(0.5*Gaussian + 0.3*Impulse + 0.2*normalized_random_walk)",
        "noise_implementation": "local fallback reproduction; not an original external generator",
        "test_noise_generation": "fixed domain/r/noise seed; sequential evaluation batches of 512",
        "metrics": "pool windows and horizons within domain; equal RW/Satellite average per seed; sample SD across seeds",
    })
    print(f"AUDIT {'PASS' if passed else 'FAIL'} | cells={len(data)} | "
          f"train/val/test windows=" + "/".join(str(len(prepared["datasets"][s])) for s in ("train", "val", "test")), flush=True)
    if not passed:
        raise RuntimeError(f"Audit failed; see {output / 'audit_report.md'}")


def train_epoch(model, loader, optimizer, cfg, device, seed, epoch):
    model.train()
    rng = np.random.default_rng(cfg.noise_seed + seed * 100_003 + epoch * 1_009)
    total, count = 0.0, 0
    for x, y in loader:
        selection = rng.random(len(x)) < cfg.train_noise_probability
        strength = np.zeros(len(x), dtype=np.float32)
        strength[selection] = rng.uniform(cfg.train_r_low, cfg.train_r_high, int(selection.sum())).astype(np.float32)
        augmented, _ = p.mixed_noise(x.numpy(), strength, rng)
        optimizer.zero_grad(set_to_none=True)
        prediction = p.normalize_prediction(model(torch.from_numpy(augmented).to(device)), cfg.pred_len)
        loss = torch.nn.functional.mse_loss(prediction, y.unsqueeze(-1).to(device))
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * len(x)
        count += len(x)
    return total / count


def fit(method, params, datasets, cfg, device, seed, directory, resume):
    directory.mkdir(parents=True, exist_ok=True)
    best_path, last_path = directory / "best.pt", directory / "last.pt"
    p.seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)
    train = DataLoader(datasets["train"], cfg.batch_size, shuffle=True, generator=generator, num_workers=0)
    val = DataLoader(datasets["val"], cfg.batch_size, shuffle=False, num_workers=0)
    model = make_model(method, params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    best, best_epoch, stale, start = math.inf, 0, 0, 1
    contract = {"method": method, "params": params, "seed": seed, "config": asdict(cfg), "device": device}
    # Reject mismatched directories even if --resume was omitted.
    contract_path = directory / "contract.json"
    if contract_path.exists() and read_json(contract_path) != json.loads(json.dumps(contract)):
        raise ValueError(f"Changed configuration in {directory}; choose a new --output directory")
    write_json(contract_path, contract)
    if resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        best, best_epoch, stale, start = state["best"], state["best_epoch"], state["stale"], state["epoch"] + 1
        generator.set_state(state["loader_rng"].cpu())
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.startswith("cuda") and state["cuda_rng"]:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
    if stale < cfg.patience:
        for epoch in range(start, cfg.max_epochs + 1):
            loss = train_epoch(model, train, optimizer, cfg, device, seed, epoch)
            validation = p._val_epoch(model, val, device, cfg.pred_len)
            if not math.isfinite(validation):
                raise RuntimeError("Non-finite validation metric")
            if validation < best:
                best, best_epoch, stale = validation, epoch, 0
                torch.save({"model": model.state_dict(), "best": best, "best_epoch": best_epoch}, best_path)
            else:
                stale += 1
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "epoch": epoch, "best": best, "best_epoch": best_epoch, "stale": stale,
                        "loader_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else []}, last_path)
            print(f"{method} seed={seed} epoch={epoch} loss={loss:.6f} val_mae_norm={validation:.6f}", flush=True)
            if stale >= cfg.patience:
                break
    state = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    return model, {"best_val_mae_norm": state["best"], "best_epoch": state["best_epoch"]}


def tune(methods, spaces, prepared, cfg, device, output, resume):
    tuned_cfg = p.Config(**asdict(cfg))
    tuned_cfg.max_epochs, tuned_cfg.patience = 80, 8
    selected, evidence = {}, []
    for method in methods:
        trials = []
        for trial, params in enumerate(spaces[method], 1):
            _, result = fit(method, params, prepared["datasets"], tuned_cfg, device, 9001,
                            output / "tuning" / method / f"trial_{trial}", resume)
            trials.append({"method": method, "trial": trial, "params": params, **result})
        best = min(trials, key=lambda item: item["best_val_mae_norm"])
        selected[method] = best["params"]
        evidence.extend(trials)
        write_json(output / "tuned_selected.json", selected)
        write_json(output / "tuning_trials.json", evidence)
    return selected


def evaluate_one(model, target, scaler, cfg, device, method, seed, fit_info, output, predictions):
    rows = []
    for domain, ds in target.items():
        for level in p.TEST_LEVELS:
            override = p._test_batch_override(cfg, domain, level)
            pred, true = p.evaluate(model, ds, scaler, device, override)
            if not np.isfinite(pred).all():
                raise RuntimeError("Non-finite test predictions")
            metrics = p.metric(true, pred)
            metrics["mape_percent"] = float(np.mean(np.abs(pred - true) / np.maximum(np.abs(true), 1e-8)) * 100)
            rows.append({"method": method, "seed": seed, "domain": domain, "noise_r": level, **metrics, **fit_info})
            if predictions:
                folder = output / "predictions"
                folder.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"battery": np.repeat([r["battery_key"] for r in ds.records], cfg.pred_len),
                              "origin_cycle_id": np.repeat([r["origin_cycle_id"] for r in ds.records], cfg.pred_len),
                              "target_cycle_id": np.asarray([r["target_cycle_ids"] for r in ds.records]).ravel(),
                              "horizon": np.tile(np.arange(1, cfg.pred_len + 1), len(ds)),
                              "y_true": true.ravel(), "y_pred": pred.ravel()}).to_csv(
                                  folder / f"{method}_seed{seed}_{domain}_r{level:g}.csv", index=False)
    return rows


def aggregate(rows, output):
    domain = pd.DataFrame(rows).sort_values(["method", "seed", "domain", "noise_r"])
    domain.to_csv(output / "results_per_domain.csv", index=False)
    metrics = ["mae", "rmse", "mape_percent"]
    seed = domain.groupby(["method", "seed", "noise_r"], as_index=False)[metrics].mean()
    seed.to_csv(output / "results_per_seed.csv", index=False)
    macro = seed.groupby(["method", "noise_r"])[metrics].agg(["mean", "std"])
    macro.columns = ["_".join(c) for c in macro.columns]
    macro.reset_index().to_csv(output / "results_macro.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="XJTU unseen-condition comparison methods")
    parser.add_argument("--action", choices=["audit", "smoke", "train", "tune"], default="audit")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(PAPER_METHODS))
    parser.add_argument("--data-root", type=Path, default=p.DATA_ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/selected.json")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2:
        parser.error("--epochs must be >=1 and --batch-size >=2")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.methods)) != len(args.methods):
        parser.error("seeds and methods must be unique")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    cfg = p.Config(model_seeds=tuple(args.seeds), batch_size=args.batch_size, max_epochs=args.epochs)
    if args.action == "smoke":
        cfg.max_epochs, cfg.batch_size, cfg.model_seeds = 1, 2, (1,)
        args.output = args.output / "smoke"
    selected = read_json(args.config)
    data, split, prepared = prepare(args.data_root, cfg)
    if args.action == "audit":
        audit(data, split, prepared, cfg, selected, args.methods, args.output)
        return
    # Lock the run to dataset location, split, model settings and numerical protocol.
    contract = {"dataset": str(args.data_root.resolve()), "split": split, "config": asdict(cfg),
                "selected": {m: selected[m] for m in args.methods}, "methods": args.methods,
                "test_levels": p.TEST_LEVELS, "device": device, "save_predictions": args.save_predictions}
    contract_path = args.output / "run_contract.json"
    if contract_path.exists() and read_json(contract_path) != json.loads(json.dumps(contract)):
        raise ValueError("Run options changed; choose a new --output directory")
    audit(data, split, prepared, cfg, selected, args.methods, args.output)
    write_json(contract_path, contract)
    if args.action == "tune":
        tune(args.methods, read_json(ROOT / "configs/search_space.json"), prepared, cfg, device, args.output, args.resume)
        print(f"TUNING COMPLETE | selected: {args.output / 'tuned_selected.json'}")
        return
    if args.action == "smoke":
        prepared["datasets"] = {s: p.WindowDataset(ds.x[:4].numpy(), ds.y[:4].numpy(), ds.records[:4])
                                for s, ds in prepared["datasets"].items() if s != "test"} | {
                                    "test": prepared["datasets"]["test"]}
    target = p._target_datasets(prepared, cfg)
    if args.action == "smoke":
        target = {c: p.WindowDataset(ds.x[:4].numpy(), ds.y[:4].numpy(), ds.records[:4]) for c, ds in target.items()}
    rows = []
    for method in args.methods:
        for seed in cfg.model_seeds:
            directory = args.output / "jobs" / method / f"seed_{seed}"
            metrics_path = directory / "metrics.json"
            if args.resume and metrics_path.exists():
                result = read_json(metrics_path)
                expected = {(c, r) for c in p.TARGET_CONDITIONS for r in p.TEST_LEVELS}
                if len(result) != 6 or {(r["domain"], r["noise_r"]) for r in result} != expected:
                    raise ValueError(f"Incomplete saved metrics: {metrics_path}")
                if args.save_predictions:
                    missing = [args.output / "predictions" / f"{method}_seed{seed}_{c}_r{r:g}.csv"
                               for c, r in expected]
                    missing = [path for path in missing if not path.exists()]
                    if missing:
                        raise FileNotFoundError(f"Missing saved predictions: {missing}; re-evaluate this job")
            else:
                model, info = fit(method, selected[method], prepared["datasets"], cfg, device, seed, directory, args.resume)
                result = evaluate_one(model, target, prepared["scaler"], cfg, device, method, seed, info,
                                      args.output, args.save_predictions)
                write_json(metrics_path, result)
            rows.extend(result)
            aggregate(rows, args.output)
    print(f"COMPLETE | {len(rows)} method/seed/domain/noise results | {args.output}", flush=True)
