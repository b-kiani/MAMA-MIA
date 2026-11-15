"""Print round-2 outputs to the run log as gzip+base64 chunks (file download from Kaggle is unreliable).

    python -m mmonc.export2 --dir /kaggle/working/round2 --extra table.csv kinetics.csv

Lines look like:  X2 <name> <part>/<parts> <payload>.   mmonc.export2.decode(log_text, out_dir) restores the files.
"""
from __future__ import annotations

import argparse
import base64
import glob
import gzip
import os
import re

CHUNK = 6000


def emit(path, name=None):
    name = name or os.path.basename(path)
    b = base64.b64encode(gzip.compress(open(path, "rb").read(), 9)).decode()
    parts = [b[i:i + CHUNK] for i in range(0, len(b), CHUNK)] or [""]
    for i, p in enumerate(parts):
        print(f"X2 {name} {i + 1}/{len(parts)} {p}", flush=True)


def decode(text: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    files = {}
    for name, i, n, payload in re.findall(r"X2 (\S+) (\d+)/(\d+) ([A-Za-z0-9+/=]*)", text):
        files.setdefault(name, {})[int(i)] = (int(n), payload)
    done = []
    for name, parts in files.items():
        n = next(iter(parts.values()))[0]
        if len(parts) != n:
            print(f"incomplete {name}: {len(parts)}/{n}")
            continue
        raw = gzip.decompress(base64.b64decode("".join(parts[i][1] for i in range(1, n + 1))))
        open(os.path.join(out_dir, name), "wb").write(raw)
        done.append(name)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    for f in sorted(glob.glob(os.path.join(a.dir, "*.csv"))) + list(a.extra):
        if os.path.exists(f):
            emit(f)
    print("X2_DONE", flush=True)


if __name__ == "__main__":
    main()
