import cv2
import numpy as np


class PostProcessor:
    def __init__(self):
        pass

    def sharpen(self, image):
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])

        return cv2.filter2D(image, -1, kernel)

    def bilateral_enhance(self, image):
        return cv2.bilateralFilter(image, 9, 75, 75)

    def gamma_correction(self, image, gamma=1.1):
        inv_gamma = 1.0 / gamma

        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in np.arange(256)
        ]).astype("uint8")

        return cv2.LUT(image, table)

    def color_balance(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        l, a, b = cv2.split(lab)

        l = cv2.equalizeHist(l)

        merged = cv2.merge((l, a, b))

        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def process(self, image):
        image = self.bilateral_enhance(image)

        image = self.sharpen(image)

        image = self.gamma_correction(image)

        image = self.color_balance(image)

        return image