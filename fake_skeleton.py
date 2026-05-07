"""
rat_movement_generator.py
──────────────────────────
Generates ~88 000 frames of synthetic rat skeleton data with 4 behaviours:
  walk, rear, groom, stretch

Each behaviour is maximally distinct in shape AND motion:
  walk:    forward locomotion, symmetric paw swing, body translates
  rear:    upright posture, front paws tucked high, static hold
  groom:   stationary, front paws oscillate rapidly toward face
  stretch: body elongates, head and tail move far apart, front paws reach forward
"""

import numpy as np
import json
import os
from collections import Counter

# ── config ────────────────────────────────────────────────────────────────────
TOTAL_FRAMES  = 88_000
FPS           = 30
NOISE_STD     = 0.5
ARENA_W       = 600
ARENA_H       = 600
START_X       = 300.0
START_Y       = 300.0

HEAD_OFFSET    = 18
TAIL_OFFSET    = 22
FRONT_PAW_X    = 10
FRONT_PAW_Y    = 10
BACK_PAW_X     = 10
BACK_PAW_Y     = -10
STEP_AMPLITUDE = 12
STEP_FREQ      = 8

BODY_PARTS = [
    "head", "center", "tail",
    "right_front_paw", "left_front_paw",
    "right_back_paw",  "left_back_paw",
]

# ── behaviour definitions ─────────────────────────────────────────────────────
# walk_speed, turn_speed, gait_scale, rear, groom, stretch
BEHAVIOUR_PARAMS = {
    "walk":    (6.0, 0.0, 1.0, False, False, False),
    "rear":    (0.0, 0.0, 0.0, True,  False, False),
    "groom":   (0.0, 0.0, 0.0, False, True,  False),
    "stretch": (0.0, 0.0, 0.0, False, False, True ),
}

DURATION_DIST = {
    #              mean  std  min   max
    "walk":    (  150,   90,  30,   500),
    "rear":    (   60,   40,  20,   200),
    "groom":   (  100,   60,  30,   300),
    "stretch": (   80,   40,  30,   200),
}

def sample_duration(rng, beh):
    mean, std, lo, hi = DURATION_DIST[beh]
    sigma2 = np.log(1 + (std / mean) ** 2)
    mu     = np.log(mean) - sigma2 / 2
    n = int(np.clip(rng.lognormal(mu, np.sqrt(sigma2)), lo, hi))
    return n


# ── sample behaviour sequence ─────────────────────────────────────────────────
print(f"Sampling behaviour sequence for {TOTAL_FRAMES:,} frames …")
rng = np.random.default_rng(42)

walk_speed_arr  = np.zeros(TOTAL_FRAMES, dtype=np.float32)
turn_speed_arr  = np.zeros(TOTAL_FRAMES, dtype=np.float32)
gait_arr        = np.zeros(TOTAL_FRAMES, dtype=np.float32)
rear_arr        = np.zeros(TOTAL_FRAMES, dtype=bool)
groom_arr       = np.zeros(TOTAL_FRAMES, dtype=bool)
stretch_arr     = np.zeros(TOTAL_FRAMES, dtype=bool)
behaviour_arr   = np.empty(TOTAL_FRAMES, dtype=object)

beh_names  = list(BEHAVIOUR_PARAMS.keys())
GT_SEGMENTS = []
fi = 0

while fi < TOTAL_FRAMES:
    prev    = GT_SEGMENTS[-1][0] if GT_SEGMENTS else None
    choices = [b for b in beh_names if b != prev]
    beh     = rng.choice(choices)
    n       = min(sample_duration(rng, beh), TOTAL_FRAMES - fi)

    ws, ts, gs, is_rear, is_groom, is_stretch = BEHAVIOUR_PARAMS[beh]

    if ts != 0.0:
        ts = float(ts) * rng.choice([-1.0, 1.0])

    sl = slice(fi, fi + n)
    walk_speed_arr[sl] = ws
    turn_speed_arr[sl] = ts
    gait_arr[sl]       = gs
    rear_arr[sl]       = is_rear
    groom_arr[sl]      = is_groom
    stretch_arr[sl]    = is_stretch
    behaviour_arr[sl]  = beh

    GT_SEGMENTS.append((beh, fi, fi + n))
    fi += n

print(f"  {len(GT_SEGMENTS):,} behaviour episodes")

# ── integrate kinematics ──────────────────────────────────────────────────────
heading_arr = np.cumsum(turn_speed_arr)

dx     = walk_speed_arr * np.cos(heading_arr)
dy     = walk_speed_arr * np.sin(heading_arr)
cx_arr = (START_X + np.cumsum(dx)) % ARENA_W
cy_arr = (START_Y + np.cumsum(dy)) % ARENA_H

dt          = 1.0 / FPS
phase_delta = np.where(gait_arr > 0, 2 * np.pi * STEP_FREQ * dt * gait_arr, 0.0)
phase_arr   = np.cumsum(phase_delta)

cos_h = np.cos(heading_arr)
sin_h = np.sin(heading_arr)

def rotate_vec(fx, fy, cos_h, sin_h):
    return fx * cos_h - fy * sin_h, fx * sin_h + fy * cos_h

cx, cy = cx_arr, cy_arr

# ── base joint offsets ────────────────────────────────────────────────────────
hx,  hy  = rotate_vec( HEAD_OFFSET, 0.0, cos_h, sin_h)
tx,  ty  = rotate_vec(-TAIL_OFFSET, 0.0, cos_h, sin_h)

rfp_bx, rfp_by = rotate_vec( FRONT_PAW_Y, -FRONT_PAW_X, cos_h, sin_h)
lfp_bx, lfp_by = rotate_vec( FRONT_PAW_Y,  FRONT_PAW_X, cos_h, sin_h)
rbp_bx, rbp_by = rotate_vec( BACK_PAW_Y,  -BACK_PAW_X,  cos_h, sin_h)
lbp_bx, lbp_by = rotate_vec( BACK_PAW_Y,   BACK_PAW_X,  cos_h, sin_h)

swing_x, swing_y = rotate_vec(STEP_AMPLITUDE * gait_arr, 0.0, cos_h, sin_h)
sin_ph  = np.sin(phase_arr)
sin_ph2 = np.sin(phase_arr + np.pi)

# ── rear posture ──────────────────────────────────────────────────────────────
REAR_HEAD_UP   = 80
REAR_CENTER_UP = 40
REAR_PAW_TUCK  = 20
REAR_TAIL_DOWN = 25
r = rear_arr.astype(np.float32)

# ── groom motion ──────────────────────────────────────────────────────────────
GROOM_FREQ     = 12
GROOM_PAW_AMP  = 14
GROOM_HEAD_BOB = 5
groom_phase    = np.cumsum(
    np.where(groom_arr, 2 * np.pi * GROOM_FREQ * dt, 0.0)
)
g          = groom_arr.astype(np.float32)
groom_sin  = np.sin(groom_phase)

# ── stretch posture ───────────────────────────────────────────────────────────
# body elongates along heading axis:
#   head extends far forward
#   tail extends far backward
#   front paws reach forward and low
#   back paws push backward
#   center stays put
# stretch builds up slowly using a smooth ramp within each episode
STRETCH_HEAD_EXT  = 35    # head extends forward along body axis
STRETCH_TAIL_EXT  = 30    # tail extends backward
STRETCH_FP_EXT    = 25    # front paws reach forward
STRETCH_BP_EXT    = 20    # back paws push back
STRETCH_FP_LOW    = 8     # front paws drop lower

# build per-frame stretch envelope: ramps up then holds
stretch_env = np.zeros(TOTAL_FRAMES, dtype=np.float32)
for label, s, e in GT_SEGMENTS:
    if label == 'stretch':
        dur      = e - s
        ramp_len = min(dur // 3, 20)   # ramp up over first third (max 20 frames)
        # ramp up
        stretch_env[s:s + ramp_len] = np.linspace(0, 1, ramp_len)
        # hold at full stretch
        stretch_env[s + ramp_len:e] = 1.0

s_env = stretch_env   # shorthand

# stretch extension vectors along heading
ext_x, ext_y = rotate_vec(1.0, 0.0, cos_h, sin_h)   # unit vector forward

# ── assemble positions ────────────────────────────────────────────────────────

# head — extends far forward during stretch
head_pos = np.stack([
    cx + hx * (1 - r) * (1 - g)
       + STRETCH_HEAD_EXT * s_env * ext_x,
    cy + hy * (1 - r) * (1 - g)
       - REAR_HEAD_UP    * r
       - GROOM_HEAD_BOB  * g * groom_sin
       + STRETCH_HEAD_EXT * s_env * ext_y,
], axis=1)

# center — stays put during stretch
center_pos = np.stack([
    cx,
    cy - REAR_CENTER_UP * r,
], axis=1)

# tail — extends far backward during stretch
tail_pos = np.stack([
    cx + tx * (1 - r)
       - STRETCH_TAIL_EXT * s_env * ext_x,
    cy + ty * (1 - r)
       + REAR_TAIL_DOWN   * r
       - STRETCH_TAIL_EXT * s_env * ext_y,
], axis=1)

# right front paw — reaches forward and low during stretch
rfp_pos = np.stack([
    cx + rfp_bx * (1 - r) * (1 - g)
       + swing_x * sin_ph  * (gait_arr > 0)
       - GROOM_PAW_AMP * g * groom_sin * cos_h
       + STRETCH_FP_EXT * s_env * ext_x,
    cy + rfp_by * (1 - r) * (1 - g)
       + swing_y * sin_ph  * (gait_arr > 0)
       - REAR_CENTER_UP * r * 0.5
       - GROOM_PAW_AMP * g * groom_sin * sin_h
       + STRETCH_FP_EXT * s_env * ext_y
       + STRETCH_FP_LOW * s_env,
], axis=1)

# left front paw
lfp_pos = np.stack([
    cx + lfp_bx * (1 - r) * (1 - g)
       - swing_x * sin_ph  * (gait_arr > 0)
       - GROOM_PAW_AMP * g * groom_sin * cos_h
       + STRETCH_FP_EXT * s_env * ext_x,
    cy + lfp_by * (1 - r) * (1 - g)
       - swing_y * sin_ph  * (gait_arr > 0)
       - REAR_CENTER_UP * r * 0.5
       - GROOM_PAW_AMP * g * groom_sin * sin_h
       + STRETCH_FP_EXT * s_env * ext_y
       + STRETCH_FP_LOW * s_env,
], axis=1)

# right back paw — pushes backward during stretch
rbp_pos = np.stack([
    cx + rbp_bx
       - swing_x * sin_ph2 * (gait_arr > 0)
       - STRETCH_BP_EXT * s_env * ext_x,
    cy + rbp_by
       - swing_y * sin_ph2 * (gait_arr > 0)
       - STRETCH_BP_EXT * s_env * ext_y,
], axis=1)

# left back paw
lbp_pos = np.stack([
    cx + lbp_bx
       + swing_x * sin_ph2 * (gait_arr > 0)
       - STRETCH_BP_EXT * s_env * ext_x,
    cy + lbp_by
       + swing_y * sin_ph2 * (gait_arr > 0)
       - STRETCH_BP_EXT * s_env * ext_y,
], axis=1)

positions = np.stack([
    head_pos, center_pos, tail_pos,
    rfp_pos,  lfp_pos,
    rbp_pos,  lbp_pos,
], axis=1).astype(np.float32)   # (T, 7, 2)

positions += rng.normal(0, NOISE_STD, positions.shape).astype(np.float32)
print(f"Positions array: {positions.shape}  dtype: {positions.dtype}")

# ── save files ────────────────────────────────────────────────────────────────
npy_path    = "rat_movement_large.npy"
labels_path = "rat_movement_large_labels.npy"
json_path   = "rat_movement_large.json"
segs_path   = "rat_movement_large_segments.json"

np.save(npy_path,    positions)
np.save(labels_path, behaviour_arr)
print(f"✓  {npy_path}      ({os.path.getsize(npy_path)/1e6:.1f} MB)")
print(f"✓  {labels_path}")

print("Building JSON annotations …")
int_pos = np.round(positions).astype(int)
annotations = {
    str(fi): {bp: int_pos[fi, j].tolist() for j, bp in enumerate(BODY_PARTS)}
    for fi in range(TOTAL_FRAMES)
}
with open(json_path, "w") as f:
    json.dump({"annotations": annotations}, f, separators=(",", ":"))
print(f"✓  {json_path}     ({os.path.getsize(json_path)/1e6:.1f} MB)")

with open(segs_path, "w") as f:
    json.dump(GT_SEGMENTS, f, indent=2)
print(f"✓  {segs_path}")

# ── summary stats ─────────────────────────────────────────────────────────────
print("\n── Behaviour breakdown ──────────────────────────────────────")
counts = Counter(behaviour_arr.tolist())
durations_by_beh = {b: [] for b in beh_names}
for label, s, e in GT_SEGMENTS:
    durations_by_beh[label].append(e - s)

for beh in beh_names:
    c    = counts.get(beh, 0)
    durs = durations_by_beh[beh]
    if durs:
        print(f"  {beh:10s}  {c:7,} frames ({c/TOTAL_FRAMES*100:.1f}%)  "
              f"n_episodes={len(durs):4d}  "
              f"dur min={min(durs):4d}  mean={np.mean(durs):6.1f}  max={max(durs):4d}")

print(f"\n  Total episodes : {len(GT_SEGMENTS):,}")
print(f"  Total frames   : {TOTAL_FRAMES:,}")