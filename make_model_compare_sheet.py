import os

import cv2
import numpy as np


ROOT = './results/model_compare'
FRAMES = ['000000', '000112', '000225', '000337', '000449']
COLUMNS = [
    ('source', 'input', '{frame}.jpg'),
    ('t', 'dehazeformer-t', '{frame}_dehazed.jpg'),
    ('s', 'dehazeformer-s', '{frame}_dehazed.jpg'),
    ('m', 'dehazeformer-m', '{frame}_dehazed.jpg'),
    ('b', 'dehazeformer-b', '{frame}_dehazed.jpg'),
]


def load_cell(path, width=220):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    h, w = image.shape[:2]
    height = int(round(h * (width / w)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def label(image, text):
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def pad_to_height(image, height):
    if image.shape[0] == height:
        return image
    pad = height - image.shape[0]
    return cv2.copyMakeBorder(image, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))


def main():
    rows = []
    for frame in FRAMES:
        cells = []
        for label_text, folder, pattern in COLUMNS:
            path = os.path.join(ROOT, folder, pattern.format(frame=frame))
            cells.append(label(load_cell(path), f'{frame} {label_text}'))

        row_height = max(cell.shape[0] for cell in cells)
        rows.append(cv2.hconcat([pad_to_height(cell, row_height) for cell in cells]))

    width = max(row.shape[1] for row in rows)
    padded_rows = []
    for row in rows:
        if row.shape[1] < width:
            row = cv2.copyMakeBorder(row, 0, 0, 0, width - row.shape[1], cv2.BORDER_CONSTANT, value=(24, 24, 24))
        padded_rows.append(row)

    sheet = cv2.vconcat(padded_rows)
    out_path = os.path.join(ROOT, 'model_comparison_sheet.jpg')
    cv2.imwrite(out_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(out_path)


if __name__ == '__main__':
    main()
