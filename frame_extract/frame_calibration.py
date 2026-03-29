import os
import shutil

from frame_extract import extract_frames_16fps


def numeric_png_sort_key(filename: str) -> int:
    name, ext = os.path.splitext(filename)
    if ext.lower() != ".png":
        return float("inf")
    return int(name)


def calibrate_extracted_frames(output_dir: str, start_frame_number: int, target_fps: int = 16) -> None:
    png_files = [f for f in os.listdir(output_dir) if f.lower().endswith(".png")]
    png_files = sorted(png_files, key=numeric_png_sort_key)

    if not png_files:
        raise ValueError(f"No PNG files found in {output_dir}")

    frame_numbers = [numeric_png_sort_key(f) for f in png_files]

    if start_frame_number not in frame_numbers:
        raise ValueError(
            f"Selected start frame {start_frame_number} not found in extracted frames.\n"
            f"Available range: {frame_numbers[0]} to {frame_numbers[-1]}"
        )

    selected_files = [f for f in png_files if numeric_png_sort_key(f) >= start_frame_number]

    ms_per_frame = 1000.0 / target_fps

    temp_dir = os.path.join(output_dir, "__tmp_calibrated__")
    os.makedirs(temp_dir, exist_ok=True)

    for f in selected_files:
        old_idx = numeric_png_sort_key(f)
        new_idx = old_idx - start_frame_number
        timestamp_ms = int(round(new_idx * ms_per_frame))

        src = os.path.join(output_dir, f)
        dst = os.path.join(temp_dir, f"{timestamp_ms}.png")
        shutil.copy2(src, dst)

    for f in png_files:
        os.remove(os.path.join(output_dir, f))

    for f in os.listdir(temp_dir):
        shutil.move(os.path.join(temp_dir, f), os.path.join(output_dir, f))

    os.rmdir(temp_dir)

    print(f"Calibration done in: {output_dir}")
    print(f"First kept frame: {start_frame_number} -> 0.png")


def main():
    video_path = "walking.mov"
    target_fps = 16

    output_folder = os.path.splitext(video_path)[0]

    extract_frames_16fps(video_path, output_folder, target_fps=target_fps)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output_frames", output_folder)

    start_frame_number = int(
        input("Enter the start frame number chosen by manual inspection: ").strip()
    )

    calibrate_extracted_frames(
        output_dir=output_dir,
        start_frame_number=start_frame_number,
        target_fps=target_fps
    )


if __name__ == "__main__":
    main()