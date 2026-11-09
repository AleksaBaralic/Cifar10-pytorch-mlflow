"""Priprema CIFAR-10 (Kaggle verzija): raspakivanje, validacija, split.

Pokretanje:
    python src/data.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import py7zr
from PIL import Image
from sklearn.model_selection import train_test_split

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
SEED = 42
TEST_SIZE = 0.10


def extract(archive: Path, out_dir: Path) -> None:
    """Raspakuje train.7z u out_dir (preskace ako je vec raspakovano)."""
    if (out_dir / "train").exists():
        print("Arhiva je vec raspakovana, preskacem.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Raspakujem {archive} (moze potrajati koji minut)...")
    with py7zr.SevenZipFile(archive) as z:
        z.extractall(out_dir)


def load_and_validate(raw_dir: Path, img_dir: Path):
    """Ucitava slike i labele i proverava njihovu ispravnost."""
    labels = pd.read_csv(raw_dir / "trainLabels.csv").sort_values("id")
    labels = labels.reset_index(drop=True)

    # Provere nad tabelom oznaka
    assert labels["id"].is_unique, "Duplirani ID-jevi u trainLabels.csv"
    assert labels.isna().sum().sum() == 0, "Postoje nedostajuce vrednosti"
    unknown = set(labels["label"]) - set(CLASSES)
    assert not unknown, f"Nepoznate klase: {unknown}"
    print(f"Broj oznaka: {len(labels)}, broj klasa: {labels['label'].nunique()}")

    # Provere nad slikama
    X = np.zeros((len(labels), 32, 32, 3), dtype=np.uint8)
    for i, img_id in enumerate(labels["id"]):
        path = img_dir / "train" / f"{img_id}.png"
        assert path.exists(), f"Nedostaje slika: {path}"
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB"))
        assert arr.shape == (32, 32, 3), f"Neocekivan oblik {arr.shape}: {path}"
        X[i] = arr
    print(f"Sve slike ucitane, oblik niza: {X.shape}")

    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    y = labels["label"].map(class_to_idx).to_numpy(dtype=np.int64)
    return labels, X, y


def make_split(labels: pd.DataFrame, y: np.ndarray, out_csv: Path) -> None:
    """Stratifikovan split: 90% trainval (za CV), 10% finalni test."""
    idx = np.arange(len(labels))
    trainval_idx, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    split = labels.copy()
    split["label_idx"] = y
    split["subset"] = "trainval"
    split.loc[test_idx, "subset"] = "test"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(out_csv, index=False)
    print(f"Split sacuvan: {out_csv} "
          f"(trainval={len(trainval_idx)}, test={len(test_idx)})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--img-dir", default="data/images")
    parser.add_argument("--processed", default="data/processed/cifar10.npz")
    parser.add_argument("--split-csv", default="configs/split_seed42.csv")
    args = parser.parse_args()

    raw_dir, img_dir = Path(args.raw_dir), Path(args.img_dir)
    extract(raw_dir / "train.7z", img_dir)
    labels, X, y = load_and_validate(raw_dir, img_dir)

    out = Path(args.processed)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y, ids=labels["id"].to_numpy())
    print(f"Sacuvano: {out}")

    make_split(labels, y, Path(args.split_csv))


if __name__ == "__main__":
    main()