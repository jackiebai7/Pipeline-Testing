"""
Batch Frame Predictor for Rat Pose Annotator
=============================================
Fill in the three paths below then just run:  python batch_predict.py

Requirements (same as annotator.py):
    pip install ultralytics opencv-python 'numpy<2'
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# EDIT THESE THREE PATHS BEFORE RUNNING
# ══════════════════════════════════════════════════════════════════════════════

MODEL_PATH    = r"C:\Users\ariAccount\runs\pose\models_iteration\final_iter\weights\best.pt"
               # e.g. r"C:\Users\You\models_iteration\rat_pose_iter_001\weights\best.pt"

VIDEO_FOLDER  = r"\\10.159.50.8\lab_common\temp_data_jackie\all_videos"
               # folder containing your .avi / .mp4 / .mov files

OUTPUT_FOLDER = r"C:\Users\ariAccount\Desktop\jackie_data_willdeletelater\actual_final_annotations"
               # where the .json annotation files will be saved
               # set to None to save them next to each video instead:
               # OUTPUT_FOLDER = None

# ── Optional settings ─────────────────────────────────────────────────────────
CONF_THRESHOLD = 0.5    # frames below this average confidence get flagged
SKIP_EXISTING  = False  # set True to skip videos that already have a .json

# ══════════════════════════════════════════════════════════════════════════════

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed. Run: pip install ultralytics")
    raise

# ── keypoint names must match the model used in annotator.py ──────────────────
KEYPOINT_NAMES = [
    'head', 'center', 'tail',
    'right_front_paw', 'left_front_paw',
    'right_back_paw', 'left_back_paw'
]

VIDEO_EXTENSIONS = {'.avi', '.mp4', '.mov', '.mkv'}


def predict_video(video_path: str, model: YOLO, conf_threshold: float = 0.5) -> dict:
    """
    Run YOLO pose prediction on every frame of a single video.

    Returns a dict of  { frame_index: { keypoint_name: (x, y) } }
    that is compatible with annotator.py's annotation format.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)

    annotations          = {}   # frame_idx -> { point_name: (x, y) }
    prediction_confidence = {}  # frame_idx -> { point_name: float }

    low_conf_count  = 0
    predicted_count = 0

    print(f"  Frames: {total_frames}  |  FPS: {fps:.1f}")

    for frame_idx in range(total_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        # ── Run YOLO prediction ───────────────────────────────────────────────
        results = model(frame, verbose=False)

        if not results or len(results) == 0:
            continue

        result = results[0]

        if not (hasattr(result, 'keypoints') and result.keypoints is not None):
            continue

        keypoints = result.keypoints

        if not (hasattr(keypoints, 'xy') and keypoints.xy.shape[0] > 0):
            continue

        # Use the first detection (highest confidence bounding box)
        kpts  = keypoints.xy[0].cpu().numpy()                              # (N_kpts, 2)
        confs = (keypoints.conf[0].cpu().numpy()
                 if hasattr(keypoints, 'conf') and keypoints.conf is not None
                 else np.ones(len(KEYPOINT_NAMES)))

        annotations[frame_idx]           = {}
        prediction_confidence[frame_idx] = {}

        avg_conf = 0.0
        for i, point_name in enumerate(KEYPOINT_NAMES):
            if i < len(kpts):
                x = int(kpts[i][0])
                y = int(kpts[i][1])
                conf = float(confs[i]) if i < len(confs) else 0.5

                annotations[frame_idx][point_name]           = [x, y]   # list for JSON
                prediction_confidence[frame_idx][point_name] = conf
                avg_conf += conf

        avg_conf /= len(KEYPOINT_NAMES)

        if avg_conf < conf_threshold:
            low_conf_count += 1

        predicted_count += 1

        # ── Progress ─────────────────────────────────────────────────────────
        if frame_idx % 100 == 0 or frame_idx == total_frames - 1:
            pct = (frame_idx + 1) / total_frames * 100
            print(f"  [{pct:5.1f}%] frame {frame_idx + 1}/{total_frames} "
                  f"| predicted so far: {predicted_count}", end='\r')

    cap.release()
    print()  # newline after \r progress

    print(f"  Predicted {predicted_count}/{total_frames} frames "
          f"| Low-confidence frames (< {conf_threshold:.2f}): {low_conf_count}")

    # ── Build the save structure ──────────────────────────────────────────────
    # annotator.py loads annotations as  { "frame_idx": { "point": [x, y] } }
    # Keys must be strings for JSON compatibility (annotator.py casts to int on load).
    save_data = {
        "annotations":            {str(k): v for k, v in annotations.items()},
        "prediction_confidence":  {str(k): v for k, v in prediction_confidence.items()},
        "video_file":             os.path.basename(video_path),
        "keypoint_names":         KEYPOINT_NAMES,
        "total_frames":           total_frames,
        "fps":                    fps,
    }

    return save_data


def main():
    # ── Validate inputs ───────────────────────────────────────────────────────
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not os.path.isdir(VIDEO_FOLDER):
        raise NotADirectoryError(f"Video folder not found: {VIDEO_FOLDER}")

    output_folder = OUTPUT_FOLDER
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    # ── Collect videos ────────────────────────────────────────────────────────
    video_files = sorted([
        f for f in Path(VIDEO_FOLDER).iterdir()
        if f.suffix.lower() in VIDEO_EXTENSIONS
    ])

    if not video_files:
        print(f"No video files found in: {VIDEO_FOLDER}")
        return

    print(f"Found {len(video_files)} video(s) in '{VIDEO_FOLDER}'")
    print(f"Model: {MODEL_PATH}")
    print(f"Confidence threshold: {CONF_THRESHOLD}")
    print()

    # ── Load model once ───────────────────────────────────────────────────────
    print("Loading model...")
    model = YOLO(MODEL_PATH)
    print("Model loaded.\n")

    # ── Process each video ────────────────────────────────────────────────────
    completed = 0
    skipped   = 0
    failed    = 0

    for idx, video_path in enumerate(video_files, start=1):
        video_path = str(video_path)
        video_name = os.path.basename(video_path)
        base_name  = os.path.splitext(video_name)[0]

        # Determine output .json path
        if output_folder:
            json_path = os.path.join(output_folder, base_name + '.json')
        else:
            json_path = os.path.join(os.path.dirname(video_path), base_name + '.json')

        print(f"[{idx}/{len(video_files)}] {video_name}")

        if SKIP_EXISTING and os.path.isfile(json_path):
            print(f"  Skipping — annotation already exists: {json_path}\n")
            skipped += 1
            continue

        try:
            save_data = predict_video(video_path, model, conf_threshold=CONF_THRESHOLD)

            with open(json_path, 'w') as f:
                json.dump(save_data, f, indent=2)

            print(f"  Saved: {json_path}\n")
            completed += 1

        except Exception as e:
            print(f"  ERROR processing {video_name}: {e}\n")
            failed += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 50)
    print(f"Done.  Completed: {completed}  |  Skipped: {skipped}  |  Failed: {failed}")
    print("=" * 50)

    if failed:
        print("\nSome videos failed. Check the error messages above.")


if __name__ == '__main__':
    main()