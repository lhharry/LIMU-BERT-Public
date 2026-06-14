'''
Compare jetson real IMU data against the Camargo public dataset, per activity
class and per IMU placement (position), in a common unit system.

Why
---
The jetson exo recordings and Camargo are two different domains (device, axis
convention, mounting). To diagnose the domain gap we line up *matching activity
classes* and *matching positions* and look at per-channel and, more importantly,
per-magnitude distributions (||acc||, ||gyro|| are invariant to axis rotation /
left-right mirroring, so they are the fairest cross-device signal).

Data sources (both under D:\\01_Code\\DATA)
------------------------------------------
* Jetson  : DATA/jetson/<HH_MM_SS>_<class>_<position>/{accelerometers,gyroscopes}.csv
            (also DATA/jetson/stair/<...>).  Each csv has columns
            Time, Left_x, Left_y, Left_z, Right_x, Right_y, Right_z  -> two
            bilateral thigh sensors that we pool together for a placement.
            Units: accel m/s^2, gyro rad/s.  "_zeroed" variants are skipped.
* Camargo : DATA/OpenSource/01_Camargo/dataset/AB*/training_data/*.csv
            thigh_{Accel,Gyro}_{X,Y,Z} + a string `Label` column.
            Units: accel g, gyro rad/s.  Position is always "thigh".

Everything is converted to a common unit system: accel -> m/s^2, gyro -> deg/s.

Class mapping
-------------
Both sides are mapped to the dense activity set used in dataset/camargo_v2.py.
Camargo raw labels (incl. transition pairs) use RAW_TO_DENSE (transition -> later
activity). Jetson class tokens (walk / rampup / jog / turnleft / turnright /
stairup / stairdown) are mapped with JETSON_TOKEN_TO_DENSE; turnleft & turnright
both collapse to "turn".  Camargo has no "jog"; jetson has no plain "stand".

Outputs (in OUT_DIR)
--------------------
* jetson_vs_camargo_stats.csv : long-format per (class, source, position,
                                channel) summary statistics.
* class_<name>.png            : per-class overlay histograms (acc x/y/z/mag,
                                gyro x/y/z/mag) for every available source.
A concise headline table (||acc|| / ||gyro|| mean +/- std) is printed.
'''

import os
import glob
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
DATA_ROOT = r"D:\01_Code\DATA"
JETSON_DIR = os.path.join(DATA_ROOT, "jetson")
CAMARGO_DIR = os.path.join(DATA_ROOT, "OpenSource", "01_Camargo", "dataset")
OUT_DIR = os.path.join(DATA_ROOT, "jetson_compare_out")

# Unit conversion to common system (accel -> m/s^2, gyro -> deg/s)
G_TO_MS2 = 9.81
RAD_TO_DEG = 57.2957795

# Cap samples kept per (camargo) class to bound memory; reservoir-subsampled.
CAMARGO_MAX_PER_CLASS = 300_000
SEED = 3431

CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
            "acc_mag", "gyro_mag"]
CHANNEL_UNIT = {c: ("m/s^2" if c.startswith("acc") else "deg/s") for c in CHANNELS}
STATS = ["mean", "std", "min", "p25", "median", "p75", "p95", "max"]

# Dense activity set (matches dataset/camargo_v2.py).
ACTIVITY_NAMES = ["stand", "walk", "turn", "jog",
                  "rampascent", "rampdescent", "stairascent", "stairdescent"]

# Camargo raw label (incl. transition pairs) -> dense activity. Transition keeps
# the later activity, identical to dataset/camargo_v2.py.
RAW_TO_DENSE = {
    "stand": "stand", "stand-walk": "walk", "walk": "walk", "walk-stand": "stand",
    "turn1": "turn", "turn2": "turn", "jog": "jog",
    "rampascent": "rampascent", "walk-rampascent": "rampascent", "rampascent-walk": "walk",
    "rampdescent": "rampdescent", "walk-rampdescent": "rampdescent", "rampdescent-walk": "walk",
    "stairascent": "stairascent", "walk-stairascent": "stairascent", "stairascent-walk": "walk",
    "stairdescent": "stairdescent", "walk-stairdescent": "stairdescent", "stairdescent-walk": "walk",
}

# Jetson folder class token -> dense activity.
JETSON_TOKEN_TO_DENSE = {
    "walk": "walk", "jog": "jog",
    "rampup": "rampascent", "rampdown": "rampdescent",
    "stairup": "stairascent", "stairdown": "stairdescent",
    "turnleft": "turn", "turnright": "turn",
}

THIGH_COLS = ["thigh_Accel_X", "thigh_Accel_Y", "thigh_Accel_Z",
              "thigh_Gyro_X", "thigh_Gyro_Y", "thigh_Gyro_Z"]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def with_magnitudes(xyz6):
    """Append ||acc|| and ||gyro|| columns to an (N,6) acc+gyro array."""
    acc_mag = np.linalg.norm(xyz6[:, 0:3], axis=1, keepdims=True)
    gyro_mag = np.linalg.norm(xyz6[:, 3:6], axis=1, keepdims=True)
    return np.hstack([xyz6, acc_mag, gyro_mag])


def summarize(arr8):
    """arr8: (N,8) -> dict channel -> {stat: value}. Empty-safe."""
    out = {}
    for j, ch in enumerate(CHANNELS):
        col = arr8[:, j] if arr8.size else np.array([])
        if col.size == 0:
            out[ch] = {s: np.nan for s in STATS}
            continue
        out[ch] = {
            "mean": float(np.mean(col)), "std": float(np.std(col)),
            "min": float(np.min(col)), "p25": float(np.percentile(col, 25)),
            "median": float(np.median(col)), "p75": float(np.percentile(col, 75)),
            "p95": float(np.percentile(col, 95)), "max": float(np.max(col)),
        }
    return out


class Reservoir:
    """Bounded, ~uniform random subsample of an unbounded row stream."""

    def __init__(self, cap, seed):
        self.cap = cap
        self.parts = []
        self.count = 0
        self.rng = np.random.default_rng(seed)

    def add(self, arr):
        if arr is None or len(arr) == 0:
            return
        self.parts.append(arr)
        self.count += len(arr)
        if self.count > self.cap * 2:
            self._compact()

    def _compact(self):
        if not self.parts:
            return
        data = np.concatenate(self.parts, axis=0)
        if len(data) > self.cap:
            idx = self.rng.choice(len(data), self.cap, replace=False)
            data = data[idx]
        self.parts = [data]
        self.count = len(data)

    def get(self):
        self._compact()
        return self.parts[0] if self.parts else np.empty((0, 6))


# -----------------------------------------------------------------------------
# Jetson loading
# -----------------------------------------------------------------------------
def iter_jetson_trials(root):
    """Yield (folder_path, class_token, position) for every jetson trial."""
    for dirpath, _dirs, files in os.walk(root):
        names = set(files)
        if "accelerometers.csv" not in names or "gyroscopes.csv" not in names:
            continue
        folder = os.path.basename(dirpath)
        toks = folder.split("_")
        if len(toks) < 5:
            print(f"  [skip] unparseable jetson folder: {folder}")
            continue
        if "zeroed" in toks[5:]:           # processed/gravity-removed variant
            print(f"  [skip] zeroed variant: {folder}")
            continue
        class_token = toks[3].lower()
        position = toks[4].lower()
        yield dirpath, class_token, position


def load_jetson_trial(folder):
    """Return (N*2, 6) array [ax,ay,az,gx,gy,gz] in common units (Left+Right pooled)."""
    acc = pd.read_csv(os.path.join(folder, "accelerometers.csv"))
    gyr = pd.read_csv(os.path.join(folder, "gyroscopes.csv"))
    n = min(len(acc), len(gyr))
    acc, gyr = acc.iloc[:n], gyr.iloc[:n]
    blocks = []
    for side in ("Left", "Right"):
        cols = [f"{side}_x", f"{side}_y", f"{side}_z"]
        a = acc[cols].to_numpy(dtype=float)              # m/s^2 already
        g = gyr[cols].to_numpy(dtype=float) * RAD_TO_DEG  # rad/s -> deg/s
        blocks.append(np.hstack([a, g]))
    return np.vstack(blocks)


def collect_jetson():
    """-> dict[(dense_class, position)] -> (N,8) array."""
    pools = {}
    print("[jetson] scanning", JETSON_DIR)
    for folder, token, position in iter_jetson_trials(JETSON_DIR):
        dense = JETSON_TOKEN_TO_DENSE.get(token)
        if dense is None:
            print(f"  [skip] unknown class token '{token}' in {os.path.basename(folder)}")
            continue
        try:
            xyz6 = load_jetson_trial(folder)
        except Exception as exc:
            print(f"  [warn] failed {os.path.basename(folder)}: {exc!r}")
            continue
        key = (dense, position)
        pools.setdefault(key, []).append(xyz6)
        print(f"  {os.path.basename(folder):32s} -> {dense:12s} pos={position:7s} "
              f"rows={len(xyz6)}")
    return {k: with_magnitudes(np.vstack(v)) for k, v in pools.items()}


# -----------------------------------------------------------------------------
# Camargo loading
# -----------------------------------------------------------------------------
def collect_camargo():
    """-> dict[(dense_class, 'thigh')] -> (N,8) array (reservoir-subsampled)."""
    files = sorted(glob.glob(os.path.join(CAMARGO_DIR, "AB*", "training_data", "*.csv")))
    print(f"[camargo] {len(files)} trial files under {CAMARGO_DIR}")
    reservoirs = {name: Reservoir(CAMARGO_MAX_PER_CLASS, SEED + i)
                  for i, name in enumerate(ACTIVITY_NAMES)}
    seen = {name: 0 for name in ACTIVITY_NAMES}
    for fi, f in enumerate(files):
        try:
            df = pd.read_csv(f, usecols=THIGH_COLS + ["Label"])
        except Exception as exc:
            print(f"  [warn] failed {os.path.basename(f)}: {exc!r}")
            continue
        sensor = df[THIGH_COLS].to_numpy(dtype=float)
        sensor[:, 0:3] *= G_TO_MS2     # g -> m/s^2
        sensor[:, 3:6] *= RAD_TO_DEG   # rad/s -> deg/s
        dense = df["Label"].map(RAW_TO_DENSE)
        for name in ACTIVITY_NAMES:
            mask = (dense == name).to_numpy()
            if mask.any():
                reservoirs[name].add(sensor[mask])
                seen[name] += int(mask.sum())
        if (fi + 1) % 500 == 0:
            print(f"  ...{fi + 1}/{len(files)} files")
    out = {}
    for name in ACTIVITY_NAMES:
        arr = reservoirs[name].get()
        if len(arr):
            out[(name, "thigh")] = with_magnitudes(arr)
            print(f"  {name:12s} total={seen[name]:>9d}  kept={len(arr):>7d}")
    return out


# -----------------------------------------------------------------------------
# Output: stats table + plots
# -----------------------------------------------------------------------------
def build_stats_table(jetson, camargo):
    rows = []
    sources = [("jetson", jetson), ("camargo", camargo)]
    for source_name, store in sources:
        for (dense, position), arr in sorted(store.items()):
            stats = summarize(arr)
            for ch in CHANNELS:
                row = {"class": dense, "source": source_name, "position": position,
                       "channel": ch, "unit": CHANNEL_UNIT[ch], "n": len(arr)}
                row.update(stats[ch])
                rows.append(row)
    cols = ["class", "source", "position", "channel", "unit", "n"] + STATS
    return pd.DataFrame(rows, columns=cols)


def plot_per_class(jetson, camargo, out_dir):
    """One overlay-histogram figure per dense class across all sources/positions."""
    store = {}
    for (dense, position), arr in jetson.items():
        store.setdefault(dense, []).append((f"jetson/{position}", arr))
    for (dense, position), arr in camargo.items():
        store.setdefault(dense, []).append((f"camargo/{position}", arr))

    for dense, series in sorted(store.items()):
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        fig.suptitle(f"jetson vs camargo — class '{dense}'  "
                     f"(accel m/s^2, gyro deg/s)", fontsize=14)
        for j, ch in enumerate(CHANNELS):
            ax = axes[j // 4, j % 4]
            for label, arr in series:
                col = arr[:, j]
                if col.size == 0:
                    continue
                lo, hi = np.percentile(col, [0.5, 99.5])
                ax.hist(col, bins=80, range=(lo, hi), density=True,
                        histtype="step", linewidth=1.6, label=label)
            ax.set_title(f"{ch} ({CHANNEL_UNIT[ch]})", fontsize=10)
            ax.grid(alpha=0.25)
            if j == 0:
                ax.legend(fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        path = os.path.join(out_dir, f"class_{dense}.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"  wrote {path}")


def print_headline(jetson, camargo):
    """Concise ||acc|| / ||gyro|| mean+/-std per class/source/position."""
    print("\n=== Headline: magnitude mean +/- std (acc m/s^2 | gyro deg/s) ===")
    header = f"{'class':12s} {'source':8s} {'position':8s} {'n':>9s}  " \
             f"{'||acc||':>16s}  {'||gyro||':>16s}"
    print(header)
    print("-" * len(header))
    rows = []
    for source_name, store in (("jetson", jetson), ("camargo", camargo)):
        for (dense, position), arr in store.items():
            rows.append((dense, source_name, position, arr))
    am = CHANNELS.index("acc_mag")
    gm = CHANNELS.index("gyro_mag")
    for dense, source_name, position, arr in sorted(rows):
        a, g = arr[:, am], arr[:, gm]
        print(f"{dense:12s} {source_name:8s} {position:8s} {len(arr):>9d}  "
              f"{a.mean():7.2f}+/-{a.std():<6.2f}  {g.mean():7.1f}+/-{g.std():<6.1f}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    jetson = collect_jetson()
    camargo = collect_camargo()

    table = build_stats_table(jetson, camargo)
    csv_path = os.path.join(OUT_DIR, "jetson_vs_camargo_stats.csv")
    table.to_csv(csv_path, index=False)
    print(f"\n[out] stats table -> {csv_path}  ({len(table)} rows)")

    print("\n[out] per-class figures:")
    plot_per_class(jetson, camargo, OUT_DIR)

    print_headline(jetson, camargo)

    j_classes = sorted({c for c, _ in jetson})
    c_classes = sorted({c for c, _ in camargo})
    print("\nclasses in jetson :", j_classes)
    print("classes in camargo:", c_classes)
    print("comparable (both) :", sorted(set(j_classes) & set(c_classes)))


if __name__ == "__main__":
    main()
