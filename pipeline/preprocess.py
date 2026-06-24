
import cv2
import numpy as np


class Preprocessor:
    def __init__(self):
        pass

    def dark_channel(self, img, size=15):
        b, g, r = cv2.split(img)
        min_img = cv2.min(cv2.min(r, g), b)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        dark = cv2.erode(min_img, kernel)

        return dark

    def estimate_atmosphere(self, img, dark):
        h, w = dark.shape
        num_pixels = h * w

        flat_dark = dark.ravel()
        flat_img = img.reshape(num_pixels, 3)

        indices = flat_dark.argsort()[-num_pixels // 1000:]

        atmosphere = np.mean(flat_img[indices], axis=0)

        return atmosphere

    def estimate_transmission(self, img, atmosphere, omega=0.95, size=15):
        normed = np.empty_like(img)

        for i in range(3):
            normed[:, :, i] = img[:, :, i] / atmosphere[i]

        dark = self.dark_channel(normed, size)

        transmission = 1 - omega * dark

        return transmission

    def recover(self, img, transmission, atmosphere, t0=0.1):
        transmission = np.clip(transmission, t0, 1)

        recovered = np.empty_like(img)

        for i in range(3):
            recovered[:, :, i] = (
                (img[:, :, i] - atmosphere[i]) / transmission
            ) + atmosphere[i]

        return np.clip(recovered, 0, 1)

    def soft_dcp_preprocess(self, image):
        img = image.astype(np.float32) / 255.0

        dark = self.dark_channel(img)

        atmosphere = self.estimate_atmosphere(img, dark)

        transmission = self.estimate_transmission(img, atmosphere)

        recovered = self.recover(img, transmission, atmosphere)

        # Blend original + recovered
        blended = cv2.addWeighted(
            img,
            0.7,
            recovered,
            0.3,
            0
        )

        blended = np.clip(blended * 255, 0, 255).astype(np.uint8)

        # CLAHE
        lab = cv2.cvtColor(blended, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        cl = clahe.apply(l)

        merged = cv2.merge((cl, a, b))

        result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        return result
