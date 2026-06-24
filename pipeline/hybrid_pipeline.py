import os
import cv2
import torch
import numpy as np

from pipeline.preprocess import Preprocessor
from pipeline.postprocess import PostProcessor
from udr.infer_udr import UDRInfer


class HybridPipeline:
    def __init__(self):
        self.preprocessor = Preprocessor()

        self.postprocessor = PostProcessor()

        self.udr = UDRInfer()

        print("Hybrid pipeline initialized")

    def run_dehazeformer(self, image):
        # PLACE YOUR REAL DEHAZEFORMER INFERENCE HERE

        # Temporary placeholder
        return image

    def process_image(self, input_path, output_path):
        image = cv2.imread(input_path)

        if image is None:
            print(f"Failed to load: {input_path}")
            return

        print("Running preprocessing...")
        preprocessed = self.preprocessor.soft_dcp_preprocess(image)

        print("Running UDR...")
        udr_output = self.udr.process(preprocessed)

        print("Running DehazeFormer...")
        dehazed = self.run_dehazeformer(udr_output)

        print("Running postprocessing...")
        final = self.postprocessor.process(dehazed)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        cv2.imwrite(output_path, final)

        print(f"Saved to: {output_path}")