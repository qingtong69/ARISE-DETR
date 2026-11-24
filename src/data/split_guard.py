"""Guard against using benchmark test annotations in model selection."""

import json
from itertools import combinations
from pathlib import Path


def check_training_splits(config):
    if config.get("eval_only"):
        raise ValueError("This configuration is for held-out evaluation. Use --test-only.")
    if "heldout_annotations" not in config:
        return

    files = {
        "train": Path(config["train_dataloader"]["dataset"]["ann_file"]),
        "validation": Path(config["val_dataloader"]["dataset"]["ann_file"]),
        "test": Path(config["heldout_annotations"]),
    }
    resolved = {split: path.resolve(strict=True) for split, path in files.items()}
    for left, right in combinations(resolved, 2):
        if resolved[left] == resolved[right]:
            raise ValueError(f"{left} and {right} resolve to the same annotation file.")

    names = {}
    for split, path in resolved.items():
        with path.open(encoding="utf-8") as stream:
            images = json.load(stream)["images"]
        names[split] = {str(image["file_name"]).replace(chr(92), "/").rsplit("/", 1)[-1]
                        for image in images}
        if not names[split]:
            raise ValueError(f"No images in the {split} annotations: {path}")

    for left, right in combinations(names, 2):
        overlap = names[left] & names[right]
        if overlap:
            example = min(overlap)
            raise ValueError(f"{left} and {right} share {len(overlap)} image names, e.g. {example}.")
