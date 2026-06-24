import cv2
import numpy as np


def _to_uint8_rgb(img):
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def _to_float_rgb(img):
    return img.astype(np.float32) / 255.0


def gray_world_white_balance(img, strength=0.45):
    means = img.reshape(-1, 3).mean(axis=0)
    gray = means.mean()
    scale = gray / np.maximum(means, 1e-6)
    balanced = np.clip(img * scale.reshape(1, 1, 3), 0, 1)
    return np.clip(img * (1.0 - strength) + balanced * strength, 0, 1)


def clahe_luminance(img, clip_limit=1.4, tile_grid_size=(8, 8)):
    rgb = _to_uint8_rgb(img)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge([l_channel, a_channel, b_channel])
    return _to_float_rgb(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def gamma_correct(img, gamma=0.95):
    return np.clip(np.power(np.clip(img, 0, 1), gamma), 0, 1)


def percentile_luminance_stretch(img, low=1.0, high=99.2, strength=0.75):
    rgb = _to_uint8_rgb(img)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_channel = lab[:, :, 0]
    lo, hi = np.percentile(l_channel, [low, high])
    if hi <= lo + 1e-6:
        return img

    stretched = (l_channel - lo) * (255.0 / (hi - lo))
    lab[:, :, 0] = np.clip(l_channel * (1.0 - strength) + stretched * strength, 0, 255)
    return _to_float_rgb(cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB))


def boost_saturation(img, factor=1.12):
    rgb = _to_uint8_rgb(img)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return _to_float_rgb(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB))


def unsharp_mask(img, amount=0.18, sigma=1.0):
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(img * (1.0 + amount) - blurred * amount, 0, 1)


def preprocess_image(img, mode):
    if mode == 'none':
        return img
    if mode != 'video':
        raise ValueError(f'Unknown preprocess mode: {mode}')

    img = gray_world_white_balance(img)
    img = clahe_luminance(img)
    img = gamma_correct(img)
    return np.clip(img, 0, 1)


def postprocess_image(img, mode):
    if mode == 'none':
        return img
    if mode != 'video':
        raise ValueError(f'Unknown postprocess mode: {mode}')

    img = percentile_luminance_stretch(img)
    img = clahe_luminance(img, clip_limit=1.15)
    img = boost_saturation(img)
    img = unsharp_mask(img)
    return np.clip(img, 0, 1)
