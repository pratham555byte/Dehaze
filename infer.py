import argparse
import os
from collections import OrderedDict

import torch

from image_conditioning import postprocess_image, preprocess_image
from models import *
from utils import chw_to_hwc, hwc_to_chw, read_img, write_img


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def load_state_dict(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)

    cleaned = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[7:]
        cleaned[key] = value

    return cleaned


def list_images(input_path):
    if os.path.isfile(input_path):
        return [input_path]

    images = []
    for name in sorted(os.listdir(input_path)):
        path = os.path.join(input_path, name)
        if os.path.isfile(path) and os.path.splitext(name.lower())[1] in IMAGE_EXTENSIONS:
            images.append(path)
    return images


def output_path_for(input_path, output_dir):
    filename = os.path.basename(input_path)
    stem, ext = os.path.splitext(filename)
    if ext.lower() not in IMAGE_EXTENSIONS:
        ext = '.png'
    return os.path.join(output_dir, f'{stem}_dehazed{ext}')


def main():
    parser = argparse.ArgumentParser(description='Run DehazeFormer on one image or a folder of images.')
    parser.add_argument('--input', required=True, help='Path to a hazy image or a folder of hazy images.')
    parser.add_argument('--output', default='./results/infer', help='Folder where dehazed images will be saved.')
    parser.add_argument('--model', default='dehazeformer-s', choices=[
        'dehazeformer-t', 'dehazeformer-s', 'dehazeformer-m', 'dehazeformer-b'
    ])
    parser.add_argument('--weights', default=None, help='Path to a .pth file. Defaults to save_models/outdoor/<model>.pth.')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--preprocess', default='none', choices=['none', 'video'])
    parser.add_argument('--postprocess', default='none', choices=['none', 'video'])
    parser.add_argument('--fp16', action='store_true', help='Use half precision (float16) for inference on CUDA.')
    args = parser.parse_args()

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

    images = list_images(args.input)
    if not images:
        raise FileNotFoundError(f'No supported images found in: {args.input}')

    os.makedirs(args.output, exist_ok=True)

    with torch.no_grad():
        for image_path in images:
            img = preprocess_image(read_img(image_path), args.preprocess)
            tensor = torch.from_numpy(hwc_to_chw(img * 2 - 1)).unsqueeze(0).to(device)
            if args.fp16 and device.type == 'cuda':
                tensor = tensor.half()

            output = network(tensor).clamp_(-1, 1)
            output = output * 0.5 + 0.5

            out_img = postprocess_image(chw_to_hwc(output.detach().cpu().squeeze(0).float().numpy()), args.postprocess)
            out_path = output_path_for(image_path, args.output)
            write_img(out_path, out_img)
            print(out_path)


if __name__ == '__main__':
    main()
