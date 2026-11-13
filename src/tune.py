"""Optimizacija hiperparametara (Optuna) sa cross-validacijom.

Primeri:
    python src/tune.py --config configs/cnn.yaml
    python src/tune.py --config configs/cnn.yaml --quick
"""
import os
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import argparse
import json
import logging
from pathlib import Path

import mlflow
import numpy as np
import optuna
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split

from train import DEFAULTS, ROOT, load_data, run_fold
from utils import get_device, set_seed

log = logging.getLogger("tune")


def suggest(trial, cfg):
    """Prostor pretrage hiperparametara."""
    params = {}
    params["optimizer"] = trial.suggest_categorical("optimizer", ["adam", "sgd"])
    if params["optimizer"] == "adam":
        params["lr"] = trial.suggest_float("lr_adam", 1e-4, 1e-2, log=True)
    else:
        params["lr"] = trial.suggest_float("lr_sgd", 1e-3, 1e-1, log=True)
    params["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256])
    params["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    model_params = dict(cfg["model_params"])
    if cfg["model"] == "cnn_bn":  # dropout postoji samo u ovom modelu
        model_params["dropout"] = trial.suggest_float("dropout", 0.1, 0.5)
    params["model_params"] = model_params
    return params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--startup", type=int, default=5,
                        help="Broj nasumicnih pokusaja pre vodjene pretrage")
    parser.add_argument("--subset", type=int, default=0,
                        help="0 = svih 45.000 trainval slika")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--quick", action="store_true",
                        help="Brz test (ne ulazi u repo)")
    args = parser.parse_args()

    cfg = {**DEFAULTS, **yaml.safe_load(Path(args.config).read_text())}
    name = cfg["name"]

    if args.quick:
        args.trials, args.startup = 2, 1
        args.subset, args.epochs, args.folds = 2000, 1, 2
        out_dir = ROOT / "results" / "quick" / "tuning" / name
        tracking = ROOT / "mlruns_quick"
        log_path = out_dir / f"tune_{name}.log"
    else:
        out_dir = ROOT / "results" / "tuning" / name
        tracking = ROOT / "mlruns"
        log_path = ROOT / "logs" / f"tune_{name}.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8"),
                  logging.StreamHandler()])
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    set_seed(cfg["seed"])
    device = get_device()
    X, y, tv_idx, mean, std = load_data()
    mean, std = mean.to(device), std.to(device)

    # Finalni test skup se ne koristi; tuning radi samo nad trainval delom
    if 0 < args.subset < len(tv_idx):
        tune_idx, _ = train_test_split(
            tv_idx, train_size=args.subset, stratify=y[tv_idx].numpy(),
            random_state=cfg["seed"])
    else:
        tune_idx = tv_idx
    y_t = y[tune_idx].numpy()
    skf = StratifiedKFold(args.folds, shuffle=True, random_state=cfg["seed"])
    splits = list(skf.split(tune_idx, y_t))
    log.info("Tuning %s | uredjaj: %s | slika: %d | %d-fold | %d epoha | "
             "%d pokusaja (%d nasumicnih)", name, device, len(tune_idx),
             args.folds, args.epochs, args.trials, args.startup)

    mlflow.set_tracking_uri(tracking.as_uri())
    mlflow.set_experiment("cifar10_tuning")

    def objective(trial):
        params = suggest(trial, cfg)
        tcfg = {**cfg, **params, "epochs": args.epochs, "n_folds": args.folds}
        accs = []
        with mlflow.start_run(run_name=f"{name}_trial{trial.number}",
                              nested=True):
            mlflow.log_params({k: (json.dumps(v) if isinstance(v, dict) else v)
                               for k, v in params.items()})
            for fold, (tr, va) in enumerate(splits, 1):
                with mlflow.start_run(run_name=f"trial{trial.number}_fold{fold}",
                                      nested=True):
                    metrics, *_ = run_fold(tcfg, fold, X, y, tune_idx[tr],
                                           tune_idx[va], mean, std, device)
                accs.append(metrics["accuracy"])
            acc = float(np.mean(accs))
            mlflow.log_metric("cv_accuracy_mean", acc)
        trial.set_user_attr("cfg_params", params)
        log.info("POKUSAJ %d: cv_acc=%.4f | %s", trial.number, acc, params)
        return acc

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=cfg["seed"], n_startup_trials=args.startup))
    with mlflow.start_run(run_name=f"{name}_tuning"):
        mlflow.log_params({"model": cfg["model"], "n_trials": args.trials,
                           "n_startup_trials": args.startup,
                           "tune_images": len(tune_idx),
                           "tune_epochs": args.epochs,
                           "tune_folds": args.folds})
        study.optimize(objective, n_trials=args.trials)
        best = study.best_trial
        best_params = best.user_attrs["cfg_params"]
        mlflow.log_metric("best_cv_accuracy", best.value)

        study.trials_dataframe().to_csv(out_dir / "trials.csv", index=False)
        (out_dir / "best_params.json").write_text(json.dumps(
            {"best_cv_accuracy": best.value, **best_params}, indent=2))
        mlflow.log_artifact(str(out_dir / "trials.csv"))
        mlflow.log_artifact(str(out_dir / "best_params.json"))

    # Nova konfiguracija: isti epochs i n_folds kao polazna, novi hiperparametri
    tuned = {**cfg, **best_params, "name": f"{name}_tuned"}
    if not args.quick:
        tuned_path = ROOT / "configs" / "tuned" / f"{name}_tuned.yaml"
        tuned_path.parent.mkdir(parents=True, exist_ok=True)
        tuned_path.write_text(yaml.safe_dump(tuned, sort_keys=False))
        log.info("Sacuvana konfiguracija: %s", tuned_path)
    log.info("GOTOVO. Najbolji cv_acc: %.4f | %s", best.value, best_params)


if __name__ == "__main__":
    main()