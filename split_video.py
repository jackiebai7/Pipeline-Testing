import cv2
import os

def split_quad_video(input_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"  Skipping (could not open): {input_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    half_w = width  // 2
    half_h = height // 2

    base_name = os.path.splitext(os.path.basename(input_path))[0]

    quadrants = {
        "top_left":     (0,      0),
        "top_right":    (half_w, 0),
        "bottom_left":  (0,      half_h),
        "bottom_right": (half_w, half_h),
    }

    writers = {
        name: cv2.VideoWriter(
            os.path.join(output_dir, f"{base_name}_{name}.mp4"),
            fourcc, fps, (half_w, half_h)
        )
        for name in quadrants
    }

    print(f"  {width}x{height} @ {fps}fps → 4x {half_w}x{half_h}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for name, (x, y) in quadrants.items():
            crop = frame[y:y+half_h, x:x+half_w]
            writers[name].write(crop)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"  Processed {frame_count} frames...")

    cap.release()
    for w in writers.values():
        w.release()

    print(f"  Done: {frame_count} frames → {output_dir}\n")


def process_folder(input_folder: str, output_folder: str):
    VIDEO_EXTENSIONS = (".mkv", ".mp4", ".avi", ".mov", ".MOV")

    videos = [
        f for f in os.listdir(input_folder)
        if f.endswith(VIDEO_EXTENSIONS)
    ]

    if not videos:
        print(f"No video files found in: {input_folder}")
        return

    print(f"Found {len(videos)} video(s) to process...\n")

    for i, filename in enumerate(videos, 1):
        input_path = os.path.join(input_folder, filename)
        print(f"[{i}/{len(videos)}] Processing: {filename}")
        split_quad_video(input_path, output_folder)

    print(f"All done! Split videos saved to: {output_folder}")


if __name__ == "__main__":
    INPUT_FOLDER  = "/path/to/your/input/folder"   # ← folder with your .mkv files
    OUTPUT_FOLDER = "/path/to/your/output/folder"  # ← where split videos will be saved

    process_folder(INPUT_FOLDER, OUTPUT_FOLDER)