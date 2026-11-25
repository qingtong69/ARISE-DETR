# ARISE-DETR

ARISE-DETR is a small-object detector based on [D-FINE](https://github.com/Peterande/D-FINE). This release provides the model source code, a model configuration, and a PyTorch inference script.

## Release scope

Datasets, annotations, training configurations, evaluation scripts, and model weights are outside the scope of this release. The example below requires a compatible checkpoint supplied by the user.

## Installation

Run from the repository root. Install matching PyTorch and torchvision builds for your environment, then install the remaining dependencies:

    python -m pip install -r requirements.txt opencv-python

## Single-image inference

The included `configs/inference/arise_visdrone.yml` describes a 10-class model. Use a checkpoint that matches this configuration. Replace `/path/to/checkpoint.pth` and `/path/to/image.jpg` with your local paths:

    python tools/inference/torch_inf.py --config configs/inference/arise_visdrone.yml --resume "/path/to/checkpoint.pth" --input "/path/to/image.jpg" --output_dir results --device cpu

The script saves an annotated image in `results/`. The example uses a separate output directory to preserve the input image.

## License

This project builds on [D-FINE](https://github.com/Peterande/D-FINE). See [LICENSE](LICENSE) for the Apache License 2.0.
