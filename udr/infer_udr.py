import cv2


class UDRInfer:
    def __init__(self):
        print("UDR initialized")

    def process(self, image):
        # PLACE YOUR REAL UDR MODEL HERE

        denoised = cv2.fastNlMeansDenoisingColored(
            image,
            None,
            5,
            5,
            7,
            21
        )

        return denoised