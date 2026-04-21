"""
Download a pre-trained English 3-gram KenLM model for use as the CPS causal proxy.

Source: OpenSLR Resource 11 — LibriSpeech LM corpus (pruned 3-gram, ~30 MB compressed).
  https://www.openslr.org/11/

The paper (Sec. 4.2) trains a 3-gram KenLM on each dataset's training split.
This script downloads a generic English model as a reasonable drop-in for general use.
For task-specific benchmarks (MT, Math, etc.) you can train your own with lmplz.

Usage:
    python scripts/download_kenlm.py [--out-dir models/kenlm]

Then pass the resulting path to generate.py:
    python scripts/generate.py --kenlm-model models/kenlm/3-gram.pruned.arpa ...
"""

import argparse
import gzip
import os
import shutil
import urllib.request
from pathlib import Path

MODEL_URL = "https://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz"
MODEL_FILENAME = "3-gram.pruned.arpa"


def download_with_progress(url: str, dest: Path) -> None:
    print(f"Downloading {url}")
    print(f"  → {dest}")

    def _reporthook(count, block_size, total_size):
        if total_size <= 0:
            print(f"\r  {count * block_size // 1024} KB", end="", flush=True)
        else:
            pct = min(100, count * block_size * 100 // total_size)
            done = pct // 2
            bar = "#" * done + "-" * (50 - done)
            mb = count * block_size / 1024 / 1024
            total_mb = total_size / 1024 / 1024
            print(f"\r  [{bar}] {pct:3d}%  {mb:.1f}/{total_mb:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_reporthook)
    print()  # newline after progress bar


def decompress_gz(src: Path, dst: Path) -> None:
    print(f"Decompressing → {dst}")
    with gzip.open(src, "rb") as f_in, open(dst, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def main():
    parser = argparse.ArgumentParser(description="Download pre-trained English KenLM model")
    parser.add_argument(
        "--out-dir",
        default="models/kenlm",
        help="Directory where the model will be saved (default: models/kenlm)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gz_path = out_dir / (MODEL_FILENAME + ".gz")
    arpa_path = out_dir / MODEL_FILENAME

    if arpa_path.exists():
        print(f"Model already exists at {arpa_path}, skipping download.")
    else:
        download_with_progress(MODEL_URL, gz_path)
        decompress_gz(gz_path, arpa_path)
        gz_path.unlink()

    size_mb = arpa_path.stat().st_size / 1024 / 1024
    print(f"\nDone. Model saved to: {arpa_path}  ({size_mb:.0f} MB)")
    print()
    print("To use with DiffuSpec:")
    print(f"  python scripts/generate.py \\")
    print(f"      --prompt 'Your prompt here' \\")
    print(f"      --kenlm-model {arpa_path} \\")
    print(f"      ...")
    print()
    print("Note: this is a generic English model. For task-specific performance matching")
    print("the paper, train a 3-gram model on your dataset's training split with:")
    print("  lmplz -o 3 < train_text.txt > lm.arpa")
    print("  build_binary lm.arpa lm.bin  # optional: faster binary format")


if __name__ == "__main__":
    main()
