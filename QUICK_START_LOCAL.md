# DehazeFormer Local Setup

This checkout is already set up with the outdoor pretrained weights:

```text
save_models/outdoor/
saved_models/outdoor/
```

The `save_models` path is used by `infer.py`. The `saved_models` path is also present because the original `test.py` defaults to that spelling.

## Daily Workflow

1. Put hazy images in `input`.
2. Run one of these commands from this folder:

```powershell
.\run_dehaze_input.bat
```

or:

```powershell
python infer.py --input .\input --output .\results --model dehazeformer-s --device auto
```

3. Collect dehazed images from `results`.

The output files are named like `<original-name>_dehazed.<ext>`.

## Dehaze One Image

From this folder:

```powershell
python infer.py --input path\to\hazy.jpg --output .\results\infer --model dehazeformer-s --device auto
```

The output image will be written as:

```text
results/infer/<original-name>_dehazed.<ext>
```

## Dehaze The Input Folder

```powershell
python infer.py --input .\input --output .\results --model dehazeformer-s --device auto
```

Supported image extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp`.

## Dehaze An Image Sequence Into Video

Put video frames in `image seq`, then run:

```powershell
.\run_dehaze_sequence.bat
```

This runs DehazeFormer independently on each frame, writes dehazed frames to:

```text
results/video_dehaze/frames/
```

and creates:

```text
results/video_dehaze/dehazed.mp4
results/video_dehaze/comparison.mp4
```

`comparison.mp4` is side-by-side source/dehazed output for checking temporal flicker. The default sequence runner uses `dehazeformer-t` and `--max-side 1280` for a quick preview. For a stronger but slower pass:

```powershell
python dehaze_sequence.py --input ".\image seq" --output ".\results\video_dehaze_s" --model dehazeformer-s --device auto --fps 24 --max-side 1280 --comparison
```

## Dehaze A Video File

For the current cycling video:

```powershell
.\run_dehaze_video.bat
```

For another video:

```powershell
python dehaze_video.py --video "path\to\video.mp4" --output-root ".\results\video_inputs" --model dehazeformer-t --device auto --max-side 1280 --comparison --preprocess video --postprocess video
```

This extracts source frames, runs DehazeFormer on each frame, then writes `dehazed.mp4` and `comparison.mp4`. Audio is not preserved in this OpenCV-based pipeline.

## Video Conditioning

For real video frames, use:

```powershell
--preprocess video --postprocess video
```

The preprocessor applies mild white balance, luminance CLAHE, and gamma conditioning before DehazeFormer. The postprocessor applies luminance stretch, mild CLAHE, saturation recovery, and light sharpening after DehazeFormer. Use `none` for raw model output:

```powershell
--preprocess none --postprocess none
```

## Model Choices

```text
dehazeformer-t  fastest, lowest memory
dehazeformer-s  good default
dehazeformer-m  stronger, slower
dehazeformer-b  strongest of the downloaded outdoor weights, slowest
```

This machine has CUDA-enabled PyTorch installed. Use `--device auto` to run on the RTX GPU when available, or `--device cpu` to force CPU.
