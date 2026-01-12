#!/usr/bin/env python3
# Copyright (c) Ruopeng Gao. All Rights Reserved.

import argparse
import os
import shutil
import subprocess
from collections import defaultdict
from configparser import ConfigParser
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: Pillow. Install with `pip install pillow` or `conda install pillow`."
    ) from exc

KNOWN_DATASETS = {
    "DanceTrack",
    "SportsMOT",
    "MOT17",
    "BFT",
    "PersonPath22_Inference",
    "CrowdHuman",
}

KNOWN_SPLITS = {"train", "val", "test"}


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize MOT tracker output.")
    parser.add_argument("--tracker-file", required=True, type=str)
    parser.add_argument("--data-root", default="./datasets", type=str)
    parser.add_argument("--dataset", default=None, type=str)
    parser.add_argument("--split", default=None, type=str)
    parser.add_argument("--sequence", default=None, type=str)
    parser.add_argument("--out-dir", default=None, type=str)
    parser.add_argument("--max-frames", default=-1, type=int)
    parser.add_argument("--include-empty", action="store_true")
    parser.add_argument("--draw-score", action="store_true")
    parser.add_argument("--line-width", default=2, type=int)
    parser.add_argument("--make-video", action="store_true")
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--video-path", default=None, type=str)
    parser.add_argument("--video-codec", default="libx264", type=str)
    return parser.parse_args()


def infer_dataset_split(tracker_file):
    parts = Path(tracker_file).parts
    dataset = next((p for p in parts if p in KNOWN_DATASETS), None)
    split = next((p for p in parts if p in KNOWN_SPLITS), None)
    return dataset, split


def load_tracker_file(tracker_file):
    frame_map = defaultdict(list)
    with open(tracker_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            frame_id = int(float(parts[0]))
            track_id = int(float(parts[1]))
            x, y, w, h = map(float, parts[2:6])
            score = float(parts[6]) if len(parts) > 6 else 1.0
            frame_map[frame_id].append((track_id, x, y, w, h, score))
    return frame_map


def infer_img_pattern(img_dir):
    if not os.path.isdir(img_dir):
        return 8, ".jpg"
    for name in sorted(os.listdir(img_dir)):
        if name.lower().endswith((".jpg", ".png")):
            stem, ext = os.path.splitext(name)
            if stem.isdigit():
                return len(stem), ext
    return 8, ".jpg"


def read_seq_length(seq_dir):
    seqinfo_path = os.path.join(seq_dir, "seqinfo.ini")
    if not os.path.isfile(seqinfo_path):
        return None
    parser = ConfigParser()
    parser.read(seqinfo_path)
    if "Sequence" not in parser:
        return None
    return int(parser["Sequence"].get("seqLength", 0))


def draw_frame(image_path, dets, draw_score=False, line_width=2):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for track_id, x, y, w, h, score in dets:
        color = id_to_color(track_id)
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = max(0, x + w), max(0, y + h)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        text = f"{track_id}"
        if draw_score:
            text = f"{track_id}:{score:.2f}"
        text_w, text_h = get_text_size(draw, text, font)
        text_x = x1
        text_y = max(0, y1 - text_h)
        draw.rectangle([text_x, text_y, text_x + text_w, text_y + text_h], fill=color)
        draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
    return image


def id_to_color(track_id):
    r = (37 * track_id) % 255
    g = (17 * track_id) % 255
    b = (29 * track_id) % 255
    r = 64 + (r % 192)
    g = 64 + (g % 192)
    b = 64 + (b % 192)
    return (int(r), int(g), int(b))


def get_text_size(draw, text, font):
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    if hasattr(draw, "textsize"):
        return draw.textsize(text, font=font)
    if hasattr(font, "getbbox"):
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    if hasattr(font, "getsize"):
        return font.getsize(text)
    return 0, 0


def main():
    args = parse_args()

    tracker_file = args.tracker_file
    if not os.path.isfile(tracker_file):
        raise FileNotFoundError(f"Tracker file not found: {tracker_file}")

    dataset = args.dataset
    split = args.split
    if dataset is None or split is None:
        inferred_dataset, inferred_split = infer_dataset_split(tracker_file)
        dataset = dataset or inferred_dataset
        split = split or inferred_split

    sequence = args.sequence or Path(tracker_file).stem
    if dataset is None or split is None:
        raise ValueError("Please provide --dataset and --split or ensure tracker path contains them.")

    seq_dir = os.path.join(args.data_root, dataset, split, sequence)
    img_dir = os.path.join(seq_dir, "img1")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(tracker_file), "visualize", sequence)
    os.makedirs(out_dir, exist_ok=True)

    frame_map = load_tracker_file(tracker_file)
    digits, ext = infer_img_pattern(img_dir)
    seq_len = read_seq_length(seq_dir)

    if args.include_empty and seq_len is not None:
        frame_ids = list(range(1, seq_len + 1))
    else:
        frame_ids = sorted(frame_map.keys())

    if args.max_frames is not None and args.max_frames > 0:
        frame_ids = frame_ids[:args.max_frames]

    for frame_id in frame_ids:
        img_name = f"{frame_id:0{digits}d}{ext}"
        img_path = os.path.join(img_dir, img_name)
        if not os.path.isfile(img_path):
            continue
        dets = frame_map.get(frame_id, [])
        image = draw_frame(
            image_path=img_path,
            dets=dets,
            draw_score=args.draw_score,
            line_width=args.line_width,
        )
        out_path = os.path.join(out_dir, img_name)
        image.save(out_path)

    print(f"Saved visualizations to: {out_dir}")

    if args.make_video:
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            raise SystemExit("ffmpeg not found. Please install ffmpeg to build a video.")
        video_path = args.video_path
        if video_path is None:
            video_path = os.path.join(out_dir, f"{sequence}.mp4")
        input_pattern = os.path.join(out_dir, f"%0{digits}d{ext}")
        cmd = [
            ffmpeg_path,
            "-y",
            "-framerate",
            str(args.fps),
            "-i",
            input_pattern,
            "-c:v",
            args.video_codec,
            "-pix_fmt",
            "yuv420p",
            video_path,
        ]
        subprocess.run(cmd, check=True)
        print(f"Saved video to: {video_path}")


if __name__ == "__main__":
    main()
