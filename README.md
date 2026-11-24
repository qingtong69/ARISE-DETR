# ARISE-DETR

ARISE-DETR (Asymmetric Reorganization and Inter-scale Semantic Enhancement DETR) is an end-to-end detector for small objects based on [D-FINE](https://github.com/Peterande/D-FINE). The encoder uses asymmetric top-down feature propagation, inter-scale semantic learning and enhancement (ISLE), partial-channel detail refinement (PCDR), and detail–semantic residual fusion (DSRF).

## Installation

The checkpoint smoke tests used Python 3.12.3, PyTorch 2.8.0, torchvision 0.23.0, and faster-coco-eval 1.7.2. The server's PyTorch build uses CUDA 12.8. Install the matching PyTorch and torchvision builds for your device, then install the remaining dependencies.

```bash
python -m pip install -r requirements.txt
```

## Data preparation

Run commands from the repository root. Configurations use paths under `data/`.

- **VisDrone2019-DET.** Download the [official data](https://github.com/VisDrone/VisDrone-Dataset) into `data/VisDrone/train` and `data/VisDrone/val`, each with its original `images` and `annotations` folders. Generate COCO JSON files with `python tools/dataset/prepare_visdrone.py data/VisDrone`.
- **CODrone-HBB.** Download [CODrone](https://github.com/AHideoKuzeA/CODrone-A-Comprehensive-Oriented-Object-Detection-benchmark-for-UAV) into `data/CODrone` with its original split directories. Run `python tools/codrone_hbb/convert_codrone_hbb.py`. The converter uses the dataset's image-level split of 5,002 training, 2,000 validation, and 3,002 test images. Different splits may include frames with the same capture filename prefix, so results describe image-level evaluation.
- **TT100K-45.** Prepare the original [TT100K](https://cg.cs.tsinghua.edu.cn/traffic-sign/) files with `python tools/dataset/prepare_tt100k_45.py /path/to/tt100k/data data/tt100k_45`. The script uses fixed benchmark class IDs and seed 42 to reserve 10% of the official training images for validation. The official test split is held out. The resulting `val.json` is an independent annotation file. Use a fresh output directory if you previously generated the old split. The script uses a symbolic link for the image root on Linux.

## Train and evaluate

| Dataset | Training configuration | Evaluation configuration |
|---|---|---|
| VisDrone2019-DET | `arise_visdrone_eval.yml` | `arise_visdrone_eval.yml` (validation benchmark) |
| CODrone-HBB | `arise_codrone_hbb.yml` | `arise_codrone_hbb_test.yml` (held-out test) |
| TT100K-45 | `arise_tt100k45.yml` | `arise_tt100k45_test.yml` (held-out test) |

The CODrone and TT100K training entry points validate the annotation paths and reject overlapping image filenames across splits. The test configurations require `--test-only` and a trained checkpoint. Use separate training output directories for new runs and select checkpoints using validation AP.

```bash
python train.py -c configs/dfine/arise_codrone_hbb.yml
python train.py -c configs/dfine/arise_codrone_hbb_test.yml -r output/arise_codrone_hbb/best.pth --test-only
python train.py -c configs/dfine/arise_tt100k45.yml
python train.py -c configs/dfine/arise_tt100k45_test.yml -r output/arise_tt100k45/best.pth --test-only
```

The archived VisDrone EMA checkpoint uses six decoder sampling points per level. Place it in `checkpoints/` and run:

```bash
python train.py -c configs/dfine/arise_visdrone_eval.yml -r checkpoints/arise_visdrone.pth --test-only
```

For a memory-limited CPU run, append `-d cpu -u use_ema=False val_dataloader.total_batch_size=1 val_dataloader.num_workers=0`. The exported model file already contains EMA weights.

## Checkpoint status

`CHECKPOINTS.json` records the archived VisDrone checkpoint and its SHA-256. The logged score at its saved epoch was 32.56 AP and 53.27 AP50. Full benchmark reproduction from the release package is pending. CODrone and TT100K test scores and checkpoints for the corrected training protocol require fresh training and evaluation.

## License

The code is based on D-FINE and retains its Apache-2.0 license and existing source attributions. See [LICENSE](LICENSE).
