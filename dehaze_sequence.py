import argparse
import os
import re
from collections import OrderedDict

import cv2
import torch

from image_conditioning import postprocess_image, preprocess_image
from infer import IMAGE_EXTENSIONS, load_state_dict
from models import *
from utils import chw_to_hwc, hwc_to_chw, read_img, write_img


def natural_key(path):
    name = os.path.basename(path)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', name)]


def list_frame_paths(input_dir):
    frames = []
    for name in os.listdir(input_dir):
        path = os.path.join(input_dir, name)
        ext = os.path.splitext(name.lower())[1]
        if os.path.isfile(path) and ext in IMAGE_EXTENSIONS:
            frames.append(path)
    return sorted(frames, key=natural_key)


def resize_if_needed(img, max_side):
    if max_side <= 0:
        return img

    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img

    new_w = max(2, int(round(w * scale)))
    new_h = max(2, int(round(h * scale)))

    # Video encoders are happier with even dimensions.
    new_w -= new_w % 2
    new_h -= new_h % 2
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def make_video(frame_paths, video_path, fps):
    first = cv2.imread(frame_paths[0], cv2.IMREAD_COLOR)
    if first is None:
        raise ValueError(f'Could not read frame for video: {frame_paths[0]}')

    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f'Could not open video writer: {video_path}')

    for path in frame_paths:
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f'Could not read frame for video: {path}')
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(frame)

    writer.release()


def make_comparison_video(source_paths, dehazed_paths, video_path, fps):
    first_source = cv2.imread(source_paths[0], cv2.IMREAD_COLOR)
    first_dehazed = cv2.imread(dehazed_paths[0], cv2.IMREAD_COLOR)
    if first_source is None or first_dehazed is None:
        raise ValueError('Could not read first source/dehazed frame for comparison video.')

    h, w = first_dehazed.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w * 2, h))
    if not writer.isOpened():
        raise RuntimeError(f'Could not open video writer: {video_path}')

    for source_path, dehazed_path in zip(source_paths, dehazed_paths):
        source = cv2.imread(source_path, cv2.IMREAD_COLOR)
        dehazed = cv2.imread(dehazed_path, cv2.IMREAD_COLOR)
        if source is None or dehazed is None:
            raise ValueError(f'Could not read comparison pair: {source_path}, {dehazed_path}')

        source = cv2.resize(source, (w, h), interpolation=cv2.INTER_AREA)
        if dehazed.shape[:2] != (h, w):
            dehazed = cv2.resize(dehazed, (w, h), interpolation=cv2.INTER_AREA)

        writer.write(cv2.hconcat([source, dehazed]))

    writer.release()


def main():
    parser = argparse.ArgumentParser(
        description='Dehaze an image sequence frame-by-frame and compile the results into a video.'
    )
    parser.add_argument('--input', default='./image seq', help='Folder containing hazy video frames.')
    parser.add_argument('--output', default='./results/video_dehaze', help='Output folder.')
    parser.add_argument('--model', default='dehazeformer-s', choices=[
        'dehazeformer-t', 'dehazeformer-s', 'dehazeformer-m', 'dehazeformer-b'
    ])
    parser.add_argument('--weights', default=None, help='Path to a .pth file. Defaults to save_models/outdoor/<model>.pth.')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--fps', default=24, type=float, help='Frames per second for the output video.')
    parser.add_argument(
        '--max-side',
        default=1280,
        type=int,
        help='Resize frames so the longest side is at most this value. Use 0 for original resolution.'
    )
    parser.add_argument('--comparison', action='store_true', help='Also write a side-by-side source/dehazed video.')
    parser.add_argument('--preprocess', default='none', choices=['none', 'video'])
    parser.add_argument('--postprocess', default='none', choices=['none', 'video'])
    parser.add_argument('--fp16', action='store_true', help='Use half precision (float16) for inference on CUDA.')
    args = parser.parse_args()

    frames = list_frame_paths(args.input)
    if not frames:
        raise FileNotFoundError(f'No supported image frames found in: {args.input}')

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'Using device: {device}')

    model_path = args.weights or os.path.join('./save_models/outdoor', args.model + '.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Weights not found: {model_path}')

    network = eval(args.model.replace('-', '_'))()
    network.load_state_dict(load_state_dict(model_path, device))
    network.to(device)
    if args.fp16 and device.type == 'cuda':
        network.half()
    network.eval()

    frame_output_dir = os.path.join(args.output, 'frames')
    os.makedirs(frame_output_dir, exist_ok=True)

    dehazed_paths = []
    with torch.no_grad():
        for index, frame_path in enumerate(frames, start=1):
            img = read_img(frame_path)
            img = resize_if_needed(img, args.max_side)
            img = preprocess_image(img, args.preprocess)
            tensor = torch.from_numpy(hwc_to_chw(img * 2 - 1)).unsqueeze(0).to(device)
            if args.fp16 and device.type == 'cuda':
                tensor = tensor.half()

            output = network(tensor).clamp_(-1, 1)
            output = output * 0.5 + 0.5

            out_img = postprocess_image(chw_to_hwc(output.detach().cpu().squeeze(0).float().numpy()), args.postprocess)
            out_name = f'{index - 1:05d}_dehazed.png'
            out_path = os.path.join(frame_output_dir, out_name)
            write_img(out_path, out_img)
            dehazed_paths.append(out_path)

            print(f'[{index}/{len(frames)}] {os.path.basename(frame_path)} -> {out_name}')

    video_path = os.path.join(args.output, 'dehazed.mp4')
    make_video(dehazed_paths, video_path, args.fps)
    print(video_path)

    if args.comparison:
        comparison_path = os.path.join(args.output, 'comparison.mp4')
        make_comparison_video(frames, dehazed_paths, comparison_path, args.fps)
        print(comparison_path)


if __name__ == '__main__':
    main()
