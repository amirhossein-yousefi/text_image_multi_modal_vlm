#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download and prepare open multimodal moderation datasets into CSVs compatible with:
 - this repo's trainers (Qwen2-VL / PaliGemma / SmolVLM)

Outputs a folder like:
  <out_root>/<dataset_name>/
    images/...
    train.csv
    val.csv
    test.csv
    class_names.txt
    LICENSE.txt (if available)

CSV schemas:
  Binary:     text,image_path,label
  Multi-label text,image_path,labels   (comma-separated labels; order = class_names.txt)
"""

import os
import json
import csv
import shutil
import argparse
from pathlib import Path
from typing import List, Dict

from tqdm import tqdm

try:
    from huggingface_hub import snapshot_download  # type: ignore
except Exception:  # pragma: no cover
    snapshot_download = None

try:
    import gdown  # type: ignore
except Exception:  # pragma: no cover
    gdown = None


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_class_names(out_dir: Path, class_names: List[str]):
    with open(out_dir / "class_names.txt", "w", encoding="utf-8") as f:
        for c in class_names:
            f.write(c + "\n")


# -----------------------------
# HATEFUL MEMES (binary)
# Source: Hugging Face dataset mirror with images + jsonl
# Structure documented in the dataset card: img/, train.jsonl, dev_seen.jsonl, test_seen.jsonl, dev_unseen.jsonl, test_unseen.jsonl
# -----------------------------

def download_and_prepare_hateful_memes(out_root: Path) -> Path:
    print("==> Downloading Hateful Memes (HF mirror) ...")
    if snapshot_download is None:
        raise RuntimeError(
            "huggingface_hub is required for hateful_memes. Install it with: pip install huggingface_hub"
        )
    # Prefer a mirror that includes images + jsonl
    repo_id = "neuralcatcher/hateful_memes"  # mirrors with img/ + jsonl
    local_repo = Path(snapshot_download(repo_id=repo_id, repo_type="dataset"))
    print(f"Downloaded snapshot to: {local_repo}")

    # Prepare output structure
    out_dir = out_root / "hateful_memes"
    images_out = out_dir / "images"
    ensure_dir(images_out)

    # Copy license if present
    for licename in ("LICENSE", "LICENSE.txt", "license.txt"):
        lic = local_repo / licename
        if lic.exists():
            ensure_dir(out_dir)
            shutil.copy2(lic, out_dir / "LICENSE.txt")
            break

    # Copy images/ (can be many small files; copytree is fine)
    src_img_dir = local_repo / "img"
    if not src_img_dir.exists():
        raise FileNotFoundError(f"Expected 'img' folder inside {local_repo}, but not found.")

    print("==> Copying images ...")
    if images_out.exists() and any(images_out.iterdir()):
        print("Images folder already populated; skipping copy.")
    else:
        shutil.copytree(src_img_dir, images_out, dirs_exist_ok=True)

    def _read_jsonl(path: Path) -> List[Dict]:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
        return rows

    # Build splits (train/dev_seen/test_seen). dev_unseen/test_unseen optional
    split_map = {
        "train.jsonl": "train.csv",
        "dev_seen.jsonl": "val.csv",
        "test_seen.jsonl": "test.csv",
    }
    # Some mirrors use {train,dev,test}.jsonl
    alt_split_map = {
        "train.jsonl": "train.csv",
        "dev.jsonl": "val.csv",
        "test.jsonl": "test.csv",
    }

    used_map = split_map
    if not all((local_repo / k).exists() for k in split_map.keys()):
        if all((local_repo / k).exists() for k in alt_split_map.keys()):
            used_map = alt_split_map
        else:
            raise FileNotFoundError("Could not find expected jsonl split files in the dataset snapshot.")

    print("==> Writing CSVs ...")
    ensure_dir(out_dir)

    # Binary task => class_names is ["harmful"]
    write_class_names(out_dir, ["harmful"])

    for src_json, dst_csv in used_map.items():
        src = local_repo / src_json
        rows = _read_jsonl(src)

        with open(out_dir / dst_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["text", "image_path", "label"])
            w.writeheader()
            for ex in rows:
                text = ex.get("text", "")
                img = ex.get("img", "")
                # HF snapshot uses relative paths like "img/123.png" -> rewrite to "images/123.png"
                img_name = Path(img).name
                image_path = f"images/{img_name}"
                label = ex.get("label", "")
                # test split may not have labels -> leave empty
                w.writerow({"text": text, "image_path": image_path, "label": label})

    print(f"✅ Hateful Memes prepared at: {out_dir}")
    print("   - class_names.txt -> 'harmful'")
    print("   - train/val/test CSVs with schema text,image_path,label")
    return out_dir


# -----------------------------
# MMHS150K (multi-label)
# Source: Authors' Google Drive link (from official project page)
# Files: MMHS150K.zip -> contains:
#   img_resized/ <id>.jpg
#   MMHS150K_GT.json (per-id metadata + 3 annotator labels)
#   splits/{train_ids.txt,val_ids.txt,test_ids.txt}
# Labels: ["NotHate","Racist","Sexist","Homophobe","Religion","OtherHate"]
# We create multi-labels by union across the 3 annotators, and DROP "NotHate"
# -----------------------------

MMHS_CLASS_SRC = ["NotHate", "Racist", "Sexist", "Homophobe", "Religion", "OtherHate"]
MMHS_CLASS_OUT = ["racist", "sexist", "homophobe", "religion", "otherhate"]  # final class names (no NotHate)


def download_and_prepare_mmhs150k(out_root: Path) -> Path:
    print("==> Downloading MMHS150K from Google Drive (large zip) ...")
    if gdown is None:
        raise RuntimeError("gdown is required for mmhs150k. Install it with: pip install gdown")
    # File id from official project Google Drive link
    file_id = "1S9mMhZFkntNnYdO-1dZXwF_8XIiFcmlF"

    raw_dir = out_root / "_raw_mmhs"
    ensure_dir(raw_dir)

    zip_path = raw_dir / "MMHS150K.zip"
    if not zip_path.exists():
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(zip_path), quiet=False)
    else:
        print("Zip already exists; skipping download.")

    extract_dir = raw_dir / "MMHS150K"
    if not extract_dir.exists() or not any(extract_dir.iterdir()):
        print("==> Extracting zip ...")
        shutil.unpack_archive(str(zip_path), str(extract_dir))
    else:
        print("Archive already extracted; skipping.")

    # Some zips extract into nested folders; search flexibly
    def find_dir(root: Path, name: str) -> Path:
        for p in root.rglob(name):
            if p.is_dir():
                return p
        raise FileNotFoundError(f"Could not find directory named '{name}' under {root}")

    def find_file(root: Path, name: str) -> Path:
        for p in root.rglob(name):
            if p.is_file():
                return p
        raise FileNotFoundError(f"Could not find file named '{name}' under {root}")

    img_dir = find_dir(extract_dir, "img_resized")
    gt_json = find_file(extract_dir, "MMHS150K_GT.json")
    split_train = find_file(extract_dir, "train_ids.txt")
    split_val = find_file(extract_dir, "val_ids.txt")
    split_test = find_file(extract_dir, "test_ids.txt")

    out_dir = out_root / "mmhs150k"
    images_out = out_dir / "images"
    ensure_dir(images_out)

    print("==> Linking/copying images ...")
    # Prefer hardlinks to save space; fall back to copy
    for src in tqdm(list(img_dir.iterdir()), desc="Images"):
        if not src.is_file():
            continue
        dst = images_out / src.name
        if dst.exists():
            continue
        try:
            os.link(src, dst)
        except Exception:
            shutil.copy2(src, dst)

    with open(gt_json, "r", encoding="utf-8") as f:
        meta = json.load(f)  # dict id -> fields

    def load_ids(path: Path) -> List[str]:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    train_ids = set(load_ids(split_train))
    val_ids = set(load_ids(split_val))
    test_ids = set(load_ids(split_test))

    ensure_dir(out_dir)
    out_cols = ["text", "image_path", "labels"]

    def labels_from_three(three: List[int]) -> List[str]:
        # three = e.g., [0,5,1]; collect all non-zero classes except NotHate (0)
        uniq = sorted(set([c for c in three if int(c) != 0]))
        out: List[str] = []
        for c in uniq:
            src_name = MMHS_CLASS_SRC[int(c)]
            if src_name == "Racist":
                out.append("racist")
            elif src_name == "Sexist":
                out.append("sexist")
            elif src_name == "Homophobe":
                out.append("homophobe")
            elif src_name == "Religion":
                out.append("religion")
            elif src_name == "OtherHate":
                out.append("otherhate")
        return out

    def write_split(ids: List[str], csv_path: Path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=out_cols)
            w.writeheader()
            for id_ in ids:
                ex = meta.get(id_)
                if ex is None:
                    continue

                text = ex.get("tweet_text", "")

                # image file is "<id>.<ext>" in img_resized; try common extensions
                img_rel = None
                for ext in (".jpg", ".jpeg", ".png"):
                    candidate = images_out / f"{id_}{ext}"
                    if candidate.exists():
                        img_rel = f"images/{candidate.name}"
                        break
                if img_rel is None:
                    continue

                labs = labels_from_three(ex.get("labels", []))
                w.writerow({
                    "text": text,
                    "image_path": img_rel,
                    "labels": ",".join(labs),
                })

    print("==> Writing CSVs ...")
    write_split(sorted(train_ids), out_dir / "train.csv")
    write_split(sorted(val_ids), out_dir / "val.csv")
    write_split(sorted(test_ids), out_dir / "test.csv")

    write_class_names(out_dir, MMHS_CLASS_OUT)

    print(f"✅ MMHS150K prepared at: {out_dir}")
    print("   - class_names.txt -> racist,sexist,homophobe,religion,otherhate")
    print("   - train/val/test CSVs with schema text,image_path,labels")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser("Download & prepare multimodal harmful-content datasets")
    ap.add_argument(
        "--dataset",
        default="mmhs150k",
        choices=["hateful_memes", "mmhs150k"],
        help="Which dataset to fetch & prepare.",
    )
    ap.add_argument("--out_root", default="data", help="Output root directory.")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    ensure_dir(out_root)

    if args.dataset == "hateful_memes":
        download_and_prepare_hateful_memes(out_root)
    elif args.dataset == "mmhs150k":
        download_and_prepare_mmhs150k(out_root)
    else:
        raise ValueError("Unsupported dataset")


if __name__ == "__main__":
    main()
