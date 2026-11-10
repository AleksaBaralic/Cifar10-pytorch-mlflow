"""Treniranje sa k-fold cross-validacijom i MLflow logovanjem.

Primeri:
    python src/train.py --config configs/cnn.yaml
    python src/train.py --config configs/cnn.yaml --quick
"""
import os
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import argparse
import json
import logging
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold

from models import build_model, count_parameters, model_size_mb
from utils import get_device, get_env_info, set_seed

ROOT = Path(__file__).resolve().parents[1]
NUM_CLASSES = 10
log = logging.getLogger("train")

DEFAULTS = {
    "model_params": {}, "epochs": 10, "batch_size": 128, "lr": 1e-3,
    "optimizer": "adam", "weight_decay": 0.0, "scheduler": "none",
    "augment": False, "n_folds": 5, "seed": 42,
}


# ----------------------------------------------------------------- podaci
def load_data():
    d = np.load(ROOT / "data/processed/cifar10.npz")
    split = pd.read_csv(ROOT / "configs/split_seed42.csv")
    assert (split["id"].to_numpy() == d["ids"]).all(), \
        "Redosled slika i splita se ne poklapa"
    norm = json.loads((ROOT / "configs/normalization.json").read_text())
    assert not np.allclose(norm["mean"], norm["mean"][0]), \
        "normalization.json izgleda pogresno (isti mean za sve kanale)"
    X = torch.from_numpy(d["X"]).permute(0, 3, 1, 2).contiguous()  # uint8
    y = torch.from_numpy(d["y"])
    mean = torch.tensor(norm["mean"], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(norm["std"], dtype=torch.float32).view(1, 3, 1, 1)
    trainval_idx = np.where(split["subset"] == "trainval")[0]
    return X, y, trainval_idx, mean, std


def prep(xb, mean, std, device):
    """uint8 batch -> normalizovan float batch."""
    return (xb.to(device).float() / 255.0 - mean) / std


def augment(x):
    """Nasumicno horizontalno okretanje + nasumicno isecanje (padding 4)."""
    b = x.size(0)
    flip = torch.rand(b, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    padded = F.pad(x, (4, 4, 4, 4), mode="reflect")
    i = torch.randint(0, 9, (b,))
    j = torch.randint(0, 9, (b,))
    out = torch.empty_like(x)
    for k in range(b):
        out[k] = padded[k, :, i[k]:i[k] + 32, j[k]:j[k] + 32]
    return out


# ------------------------------------------------------------- evaluacija
@torch.no_grad()
def predict(model, X, idx, mean, std, device, bs=1000):
    model.eval()
    outs = []
    for s in range(0, len(idx), bs):
        b = torch.from_numpy(idx[s:s + bs])
        outs.append(model(prep(X[b], mean, std, device)).cpu())
    return torch.cat(outs)


def compute_metrics(y_true, probs):
    pred = probs.argmax(1)
    return {
        "accuracy": float((pred == y_true).mean()),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "macro_precision": float(precision_score(
            y_true, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, pred, average="macro")),
        "roc_auc_ovr": float(roc_auc_score(
            y_true, probs, multi_class="ovr", average="macro")),
        "pr_auc_macro": float(average_precision_score(
            np.eye(NUM_CLASSES)[y_true], probs, average="macro")),
    }


def make_optimizer(cfg, model):
    if cfg["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                weight_decay=cfg["weight_decay"])
    if cfg["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9,
                               weight_decay=cfg["weight_decay"], nesterov=True)
    raise ValueError(f"Nepoznat optimizer: {cfg['optimizer']}")


# ------------------------------------------------------------------- fold
def run_fold(cfg, fold, X, y, tr_idx, va_idx, mean, std, device):
    set_seed(cfg["seed"] + fold)
    model = build_model(cfg["model"], **cfg["model_params"]).to(device)
    opt = make_optimizer(cfg, model)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
             if cfg["scheduler"] == "cosine" else None)

    proc = psutil.Process()
    psutil.cpu_percent(None)  # pocetak merenja CPU-a
    peak_ram = proc.memory_info().rss
    y_va = y[va_idx]
    history, train_time = [], 0.0

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.perf_counter()
        model.train()
        perm = tr_idx[torch.randperm(len(tr_idx)).numpy()]
        total, n = 0.0, 0
        for s in range(0, len(perm), cfg["batch_size"]):
            b = torch.from_numpy(perm[s:s + cfg["batch_size"]])
            xb = prep(X[b], mean, std, device)
            yb = y[b].to(device)
            if cfg["augment"]:
                xb = augment(xb)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
            n += len(b)
        lr_now = opt.param_groups[0]["lr"]
        if sched:
            sched.step()
        epoch_time = time.perf_counter() - t0
        train_time += epoch_time

        logits = predict(model, X, va_idx, mean, std, device)
        val_loss = F.cross_entropy(logits, y_va).item()
        val_acc = (logits.argmax(1) == y_va).float().mean().item()
        peak_ram = max(peak_ram, proc.memory_info().rss)

        row = {"fold": fold, "epoch": epoch, "train_loss": total / n,
               "val_loss": val_loss, "val_acc": val_acc, "lr": lr_now,
               "epoch_time_s": epoch_time}
        history.append(row)
        mlflow.log_metrics({k: v for k, v in row.items()
                            if k not in ("fold", "epoch")}, step=epoch)
        log.info("fold %d epoha %2d/%d | train_loss %.4f | val_loss %.4f | "
                 "val_acc %.4f | %.1fs", fold, epoch, cfg["epochs"],
                 row["train_loss"], val_loss, val_acc, epoch_time)

    cpu_util = psutil.cpu_percent(None)

    # vreme inferencije na validacionom delu (finalni model)
    t0 = time.perf_counter()
    logits = predict(model, X, va_idx, mean, std, device)
    inf_ms = (time.perf_counter() - t0) / len(va_idx) * 1000

    probs = F.softmax(logits.double(), dim=1).numpy()
    y_np = y_va.numpy()
    metrics = compute_metrics(y_np, probs)
    metrics["val_loss"] = F.cross_entropy(logits, y_va).item()
    metrics.update({
        "train_time_s": train_time,
        "inference_ms_per_image": inf_ms,
        "peak_ram_mb": peak_ram / 1024 ** 2,
        "cpu_util_percent": cpu_util,
    })
    mlflow.log_metrics(metrics)
    cm = confusion_matrix(y_np, probs.argmax(1), labels=list(range(NUM_CLASSES)))
    info = {"n_params": count_parameters(model),
            "model_size_mb": model_size_mb(model)}
    return metrics, history, probs, cm, info


# ------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--quick", action="store_true",
                        help="Brz test na malom podskupu (ne ulazi u repo)")
    args = parser.parse_args()

    cfg = {**DEFAULTS, **yaml.safe_load(Path(args.config).read_text())}
    name = cfg["name"]

    if args.quick:
        out_dir = ROOT / "results" / "quick" / name
        tracking = ROOT / "mlruns_quick"
        cfg["epochs"] = min(cfg["epochs"], 2)
        cfg["n_folds"] = 2
    else:
        out_dir = ROOT / "results" / "runs" / name
        tracking = ROOT / "mlruns"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = (out_dir / f"{name}.log" if args.quick
                else ROOT / "logs" / f"{name}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8"),
                  logging.StreamHandler()])

    set_seed(cfg["seed"])
    device = get_device()
    X, y, tv_idx, mean, std = load_data()
    if args.quick:
        rng = np.random.default_rng(cfg["seed"])
        tv_idx = np.sort(rng.choice(tv_idx, 3000, replace=False))
    mean, std = mean.to(device), std.to(device)
    log.info("Konfiguracija: %s | uredjaj: %s | slika za CV: %d",
             cfg, device, len(tv_idx))

    env = get_env_info()
    mlflow.set_tracking_uri(tracking.as_uri())
    mlflow.set_experiment("cifar10")

    skf = StratifiedKFold(cfg["n_folds"], shuffle=True,
                          random_state=cfg["seed"])
    y_tv = y[tv_idx].numpy()
    oof_probs = np.zeros((len(tv_idx), NUM_CLASSES), dtype=np.float32)
    oof_fold = np.zeros(len(tv_idx), dtype=np.int64)
    fold_rows, history_rows = [], []

    with mlflow.start_run(run_name=name):
        flat = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                for k, v in cfg.items()}
        mlflow.log_params(flat)
        mlflow.log_param("torch_threads", torch.get_num_threads())
        mlflow.set_tags({"git_commit": env["git_commit"],
                         "torch": env["packages"]["torch"],
                         "quick": str(args.quick)})
        mlflow.log_dict(env, "env_info.json")

        for fold, (tr_pos, va_pos) in enumerate(skf.split(tv_idx, y_tv), 1):
            with mlflow.start_run(run_name=f"{name}_fold{fold}", nested=True):
                metrics, hist, probs, cm, info = run_fold(
                    cfg, fold, X, y, tv_idx[tr_pos], tv_idx[va_pos],
                    mean, std, device)
            oof_probs[va_pos] = probs
            oof_fold[va_pos] = fold
            fold_rows.append({"fold": fold, **metrics})
            history_rows += hist
            pd.DataFrame(cm).to_csv(out_dir / f"confusion_matrix_fold{fold}.csv",
                                    index=False)
            log.info("FOLD %d gotov: acc=%.4f, f1=%.4f, vreme=%.1fs",
                     fold, metrics["accuracy"], metrics["macro_f1"],
                     metrics["train_time_s"])

        fold_df = pd.DataFrame(fold_rows)
        metric_cols = [c for c in fold_df.columns if c != "fold"]
        summary = {"name": name, "n_params": info["n_params"],
                   "model_size_mb": info["model_size_mb"]}
        for c in metric_cols:
            summary[f"{c}_mean"] = float(fold_df[c].mean())
            summary[f"{c}_std"] = float(fold_df[c].std(ddof=1))
        summary["total_train_time_s"] = float(fold_df["train_time_s"].sum())
        mlflow.log_metrics({k: v for k, v in summary.items()
                            if isinstance(v, float)})
        mlflow.log_metrics({"n_params": info["n_params"],
                            "model_size_mb": info["model_size_mb"]})

        fold_df.to_csv(out_dir / "fold_metrics.csv", index=False)
        pd.DataFrame(history_rows).to_csv(out_dir / "history.csv", index=False)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        (out_dir / "env_info.json").write_text(json.dumps(env, indent=2))
        (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
        np.savez_compressed(out_dir / "oof_predictions.npz", idx=tv_idx,
                            y=y_tv, probs=oof_probs, fold=oof_fold)
        for f in out_dir.iterdir():
            if f.name != "oof_predictions.npz" and f.suffix != ".log":
                mlflow.log_artifact(str(f))

    log.info("GOTOVO. Accuracy: %.4f +/- %.4f | ukupno vreme treniranja: %.1fs",
             summary["accuracy_mean"], summary["accuracy_std"],
             summary["total_train_time_s"])


if __name__ == "__main__":
    main()