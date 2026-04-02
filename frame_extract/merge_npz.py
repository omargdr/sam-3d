import numpy as np
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
SESSION_NAME = "walking"
TARGET_FPS   = 16

JOINTS_FOLDER = Path(f"outputs/joints_npz/{SESSION_NAME}")
OUTPUT_PATH   = Path(f"outputs/sessions_final/{SESSION_NAME}_joints.npz")
# ────────────────────────────────────────────────────────────────────────────

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

npz_files = sorted(JOINTS_FOLDER.glob("*.npz"), key=lambda p: int(p.stem))

if not npz_files:
    raise SystemExit(f"No NPZ files found in {JOINTS_FOLDER}")

first = np.load(npz_files[0], allow_pickle=True)
print(f"Keys in NPZ: {list(first.keys())}")
for k, v in first.items():
    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

joint_names = first["joint_names"]

joints_list   = []
timestamps_ms = []

for npz_file in npz_files:
    frame_index = int(npz_file.stem)
    data = np.load(npz_file, allow_pickle=True)

    points = data["person_0"]
    if points.ndim == 3:
        points = points[0]

    joints_list.append(points)   # shape (30, 3)
    timestamps_ms.append(frame_index * (1000 / TARGET_FPS))

joints        = np.stack(joints_list, axis=0)
timestamps_ms = np.array(timestamps_ms, dtype=np.float64)

print(f"\nSession:             {SESSION_NAME}")
print(f"joints shape:        {joints.shape}")
print(f"timestamps_ms shape: {timestamps_ms.shape}")
print(f"Duration:            {timestamps_ms[-1] / 1000:.1f}s ({len(npz_files)} frames)")

np.savez_compressed(
    OUTPUT_PATH,
    joints=joints,
    timestamps_ms=timestamps_ms,
    joint_names=joint_names,
)
print(f"Saved to {OUTPUT_PATH}")