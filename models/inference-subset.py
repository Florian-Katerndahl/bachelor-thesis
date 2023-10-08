#!/usr/bin/env python
# coding: utf-8

import glob
import warnings

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib as plt

from sklearn.metrics import recall_score, jaccard_score

import pickle

import torch
from fastai.vision.all import *
from fastai.data.all import *
from fastai.data.transforms import *

# Setup

synthetic_datasets = [(s.split("/")[-1].replace("_", "-"), s) for s in glob.glob("./data/experiment_[1-8]")]
synthetic_datasets = {k: {"path": Path(v)} for k, v in synthetic_datasets if glob.glob(v + "/**/*")}

image_classes: List[str] = ["other", "tree"]


def get_these_ys(p: Path):
    id, *split_fname = p.stem.split("_")
    f = p.parent.parent / "valid" / f"{id}_gt_{'_'.join(split_fname)}{p.suffix}"
    msk = np.array(Image.open(f))
    return PILMask.create(msk)


syntrees = DataBlock(
    blocks=(ImageBlock, MaskBlock(image_classes)),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.3),
    get_y=get_these_ys,
    item_tfms=RandomResizedCrop((256, 256)),
    batch_tfms=[Dihedral(), Brightness(max_lighting=0.4), Contrast(), Saturation(0.8)],
)

syntrees_inf = DataBlock(
    blocks=(ImageBlock, MaskBlock(image_classes)),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.3),
    get_y=get_these_ys,
    # item_tfms=RandomResizedCrop((256,256)),
    batch_tfms=[Dihedral(), Brightness(max_lighting=0.4), Contrast(), Saturation(0.8)],
)


def sensitivity(p: torch.tensor, t: torch.tensor) -> float:
    return recall_score(np.array(t).ravel(), np.array(p).ravel(), zero_division=0, pos_label=1)


def specificity(p: torch.tensor, t: torch.tensor) -> float:
    return recall_score(np.array(t).ravel(), np.array(p).ravel(), zero_division=0, pos_label=0)


def avg_sensitivity(truths: List[torch.tensor], predictions: List[torch.tensor]) -> float:
    assert len(truths) == len(predictions)
    added_sensitivities: List[float] = []
    for pred, truth in zip(predictions, truths):
        added_sensitivities.append(sensitivity(pred, truth))
    return np.nanmean(added_sensitivities)


def avg_specificity(truths: List[torch.tensor], predictions: List[torch.tensor]) -> float:
    assert len(truths) == len(predictions)
    added_specificities: List[float] = []
    for pred, truth in zip(predictions, truths):
        added_specificities.append(specificity(pred, truth))
    return np.nanmean(added_specificities)


def dice(p: torch.tensor, t: torch.tensor) -> float:
    jac = jaccard_score(np.array(t).ravel(), np.array(p).ravel(), zero_division=1.0)
    return (2.0 * jac) / (1.0 + jac)


def NicerDicer(truths: List[torch.tensor], predictions: List[torch.tensor]) -> float:
    assert len(truths) == len(predictions)
    added_dices: List[float] = []
    for pred, truth in zip(predictions, truths):
        added_dices.append(dice(pred, truth))
    return np.nanmean(added_dices)


# https://benjaminwarner.dev/2021/10/01/inference-with-fastai
# https://medium.com/mlearning-ai/understanding-evaluation-metrics-in-medical-image-segmentation-d289a373a3f
def inference(
    ds: Path,
    datasets: Optional[Dict[str, str]] = None,
    exp: Optional[str] = None,
    long_path: Optional[Path] = None,
    mod_arch=models.resnet34,
) -> Tuple[List[TensorBase], float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)  # ignore known UserWarnings about pretrained models
        dls: DataLoaders = syntrees_inf.dataloaders(
            ds / "real", bs=8
        )  # pseudo data set; not used; batch size could be anything but acutally does have an influence
        dls.vocab = image_classes
        dls.c = len(image_classes)

        inference_learner = unet_learner(dls, mod_arch, normalize=True, pretrained=True, metrics=Dice)
        inference_learner.load(datasets.get(exp).get("model_path") if datasets and exp else long_path, device="cuda")

        input_paths: L = get_image_files(ds, folders="real")
        dls = inference_learner.dls.test_dl(input_paths, with_decoded=True)
        mdls = [
            cast(np.array(get_these_ys(f), np.float64), TensorBase) for f in input_paths
        ]  # ehmm... get validation masks and convert to tensors

        preds, _ = inference_learner.get_preds(dl=dls)
        flattened_preds = [preds[i].argmax(dim=0) for i in range(0, preds.shape[0])]

        test_avg_dice = NicerDicer(mdls, flattened_preds)
        test_avg_sensitivity = avg_sensitivity(mdls, flattened_preds)
        test_avg_specificity = avg_specificity(mdls, flattened_preds)

        return flattened_preds, test_avg_dice, test_avg_sensitivity, test_avg_specificity, input_paths


# Testing/ Inference

## Experiment 1
flat_preds_1_resnet34, dice_metric, sens, speci, test_paths_1_resnet34 = inference(
    long_path=Path("./experiment-1-resnet34"), ds=Path("./data/validation/splitted-test"), mod_arch=models.resnet34
)
print(
    f"Model 1 (ResNet34) achieved a dice accuracy of {round(dice_metric, 4)}, sensitivity of {round(sens, 4)}, specificity of {round(speci, 4)}"
)

flat_preds_1_resnet50, dice_metric, sens, speci, test_paths_1_resnet50 = inference(
    long_path=Path("./experiment-1-resnet50"), ds=Path("./data/validation/splitted-test"), mod_arch=models.resnet50
)
print(
    f"Model 1 (ResNet50) achieved a dice accuracy of {round(dice_metric, 4)}, sensitivity of {round(sens, 4)}, specificity of {round(speci, 4)}"
)

## Experiment 2
flat_preds_2_resnet34, dice_metric, sens, speci, test_paths_2_resnet34 = inference(
    long_path=Path("./experiment-2-resnet34"), ds=Path("./data/validation/splitted-test"), mod_arch=models.resnet34
)
print(
    f"Model 2 (ResNet34) achieved a dice accuracy of {round(dice_metric, 4)}, sensitivity of {round(sens, 4)}, specificity of {round(speci, 4)}"
)

flat_preds_2_resnet50, dice_metric, sens, speci, test_paths_2_resnet50 = inference(
    long_path=Path("./experiment-2-resnet50"), ds=Path("./data/validation/splitted-test"), mod_arch=models.resnet50
)
print(
    f"Model 2 (ResNet50) achieved a dice accuracy of {round(dice_metric, 4)}, sensitivity of {round(sens, 4)}, specificity of {round(speci, 4)}"
)

## Experiment 3
flat_preds_3_resnet34, dice_metric, sens, speci, test_paths_3_resnet34 = inference(
    long_path=Path("./experiment-3-resnet34"), ds=Path("./data/validation/splitted-test"), mod_arch=models.resnet34
)
print(
    f"Model 3 (ResNet34) achieved a dice accuracy of {round(dice_metric, 4)}, sensitivity of {round(sens, 4)}, specificity of {round(speci, 4)}"
)

flat_preds_3_resnet50, dice_metric, sens, speci, test_paths_3_resnet50 = inference(
    long_path=Path("./experiment-3-resnet50"), ds=Path("./data/validation/splitted-test"), mod_arch=models.resnet50
)
print(
    f"Model 3 (ResNet50) achieved a dice accuracy of {round(dice_metric, 4)}, sensitivity of {round(sens, 4)}, specificity of {round(speci, 4)}"
)

## Experiment 4
flat_preds_4_resnet34, dice_metric, sens, speci, test_paths_4_resnet34 = inference(
    long_path=Path("./experiment-4-resnet34"), ds=Path("./data/validation/splitted-test"), mod_arch=models.resnet34
)
print(
    f"Model 4 (ResNet34) achieved a dice accuracy of {round(dice_metric, 4)}, sensitivity of {round(sens, 4)}, specificity of {round(speci, 4)}"
)

flat_preds_4_resnet50, dice_metric, sens, speci, test_paths_4_resnet50 = inference(
    long_path=Path("./experiment-4-resnet50"), ds=Path("./data/validation/splitted-test"), mod_arch=models.resnet50
)
print(
    f"Model 4 (ResNet50) achieved a dice accuracy of {round(dice_metric, 4)}, sensitivity of {round(sens, 4)}, specificity of {round(speci, 4)}"
)

## Experiment 5
flat_preds_5_resnet34, dice_metric, sens, speci, test_paths_5_resnet34 = inference(
     long_path=Path("./experiment-5-resnet34"), ds=Path("./data/validation/splitted-test"), mod_arch=models.resnet34
)
print(
    f"Model 5 (ResNet34) achieved a dice accuracy of {round(dice_metric, 4)}, sensitivity of {round(sens, 4)}, specificity of {round(speci, 4)}"
)

flat_preds_5_resnet50, dice_metric, sens, speci, test_paths_5_resnet50 = inference(
    long_path=Path("./experiment-5-resnet50"), ds=Path("./data/validation/splitted-test"), mod_arch=models.resnet50
)
print(
    f"Model 5 (ResNet50) achieved a dice accuracy of {round(dice_metric, 4)}, sensitivity of {round(sens, 4)}, specificity of {round(speci, 4)}"
)
