"""
Video Frame Mining Utility for DialReader
Extracts sparse keyframes from the dial adjustment video (max ~100 frames)
to provide varied drum orientations and mid-roll transitions.
Rule: These frames are designated strictly for the training set (never val/test).
"""

import argparse
import sys
from pathlib import Path
import cv2
from tqdm import tqdm


def mine_frames(
    video_path: Path,
    output_dir: Path,
    stride: int = 30,
    max_frames: int = 100,
    prefix: str = "knife_vid_"
) -> list[Path]:
    """
    Extract sparse frames from the dial video.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Opening video: {video_path.name}")
    print(f"Total frames: {total_frames}, FPS: {fps:.2f}, Stride: {stride}")

    # Calculate actual stride if total_frames / stride > max_frames
    if total_frames // stride > max_frames:
        stride = total_frames // max_frames
        print(f"Adjusted stride to {stride} to stay under max_frames={max_frames}")

    saved_paths = []
    frame_idx = 0
    pbar = tqdm(total=min(total_frames // stride, max_frames), desc="Mining frames")

    while cap.isOpened() and len(saved_paths) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        out_name = f"{prefix}f{frame_idx:05d}.jpg"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved_paths.append(out_path)
        pbar.update(1)

        frame_idx += stride

    pbar.close()
    cap.release()
    print(f"Extracted {len(saved_paths)} video frames to {output_dir}")
    return saved_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mine sparse frames from dial video")
    parser.add_argument(
        "--video_path",
        type=str,
        default=r"C:\Users\ABHI\Downloads\Dial data extraction\anish\Knife\video\20261006_084248000_iOS.MOV",
        help="Path to source MOV/MP4 video"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"c:\Users\ABHI\OneDrive\Desktop\DialReader\dial-reader\data\raw\knife_video_frames",
        help="Directory to save extracted frame images"
    )
    parser.add_argument("--stride", type=int, default=30, help="Frame step")
    parser.add_argument("--max_frames", type=int, default=100, help="Maximum frames to extract")

    args = parser.parse_args()
    mine_frames(Path(args.video_path), Path(args.output_dir), args.stride, args.max_frames)
