import argparse
import os

import cv2

from dehaze_sequence import list_frame_paths, main as dehaze_sequence_main


def safe_name(path):
    name = os.path.splitext(os.path.basename(path))[0]
    keep = []
    for char in name:
        if char.isalnum():
            keep.append(char.lower())
        elif char in [' ', '-', '_']:
            keep.append('_')
    cleaned = ''.join(keep).strip('_')
    while '__' in cleaned:
        cleaned = cleaned.replace('__', '_')
    return cleaned or 'video'


def extract_frames(video_path, output_dir, image_ext='.jpg', quality=95):
    os.makedirs(output_dir, exist_ok=True)

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise FileNotFoundError(f'Could not open video: {video_path}')

    fps = capture.get(cv2.CAP_PROP_FPS) or 24
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    index = 0
    written = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        out_path = os.path.join(output_dir, f'{index:06d}{image_ext}')
        if image_ext.lower() in ['.jpg', '.jpeg']:
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            cv2.imwrite(out_path, frame)

        index += 1
        written += 1
        if written % 25 == 0:
            print(f'Extracted {written}/{frame_count or "?"} frames')

    capture.release()
    print(f'Extracted {written} frames at {width}x{height}, {fps:.3f} FPS')
    return fps, written


def main():
    parser = argparse.ArgumentParser(description='Extract a video, dehaze frames, and compile dehazed video output.')
    parser.add_argument('--video', required=True, help='Input hazy video file.')
    parser.add_argument('--output-root', default='./results/video_inputs', help='Root folder for video outputs.')
    parser.add_argument('--model', default='dehazeformer-t', choices=[
        'dehazeformer-t', 'dehazeformer-s', 'dehazeformer-m', 'dehazeformer-b'
    ])
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--max-side', default=1280, type=int, help='Use 0 for original resolution.')
    parser.add_argument('--comparison', action='store_true', help='Write a side-by-side source/dehazed video.')
    parser.add_argument('--preprocess', default='video', choices=['none', 'video', 'clahe'])
    parser.add_argument('--postprocess', default='video', choices=['none', 'video', 'sharpen'])
    parser.add_argument('--fp16', action='store_true', help='Use half precision (float16) for inference on CUDA.')
    args = parser.parse_args()

    output_dir = os.path.join(args.output_root, safe_name(args.video))
    source_frames_dir = os.path.join(output_dir, 'source_frames')
    dehaze_output_dir = os.path.join(output_dir, 'dehazed_output')

    existing_frames = list_frame_paths(source_frames_dir) if os.path.isdir(source_frames_dir) else []
    if existing_frames:
        capture = cv2.VideoCapture(args.video)
        fps = capture.get(cv2.CAP_PROP_FPS) or 24
        capture.release()
        print(f'Using existing extracted frames in {source_frames_dir}')
    else:
        fps, _ = extract_frames(args.video, source_frames_dir)

    dehaze_args = [
        'dehaze_sequence.py',
        '--input', source_frames_dir,
        '--output', dehaze_output_dir,
        '--model', args.model,
        '--device', args.device,
        '--fps', str(fps),
        '--max-side', str(args.max_side),
        '--preprocess', args.preprocess,
        '--postprocess', args.postprocess,
    ]
    if args.comparison:
        dehaze_args.append('--comparison')
    if args.fp16:
        dehaze_args.append('--fp16')

    import sys
    old_argv = sys.argv
    try:
        sys.argv = dehaze_args
        dehaze_sequence_main()
    finally:
        sys.argv = old_argv

    print(f'Dehazed video: {os.path.join(dehaze_output_dir, "dehazed.mp4")}')
    if args.comparison:
        print(f'Comparison video: {os.path.join(dehaze_output_dir, "comparison.mp4")}')


if __name__ == '__main__':
    main()
