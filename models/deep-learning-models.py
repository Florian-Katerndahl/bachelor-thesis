#!/usr/bin/env python
# coding: utf-8

import glob
import warnings

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib as plt

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


class KekCallback(Callback):
    "Shamelessly stolen from ShowGraphCallback"
    order, run_valid = 65, False

    def before_fit(self):
        self.run = not hasattr(self.learn, "lr_finder") and not hasattr(self, "gather_preds")
        if not (self.run):
            return
        self.nb_batches = []
        self.validation_dices = []
        self.res = {
            "stop": None,
            "batch_items": None,
            "arch": None,
            "batch_size": None,
            "initial_learning_rate": None,
            "mini_batch": None,
            "train_loss": None,
            "validation_idxs": None,
            "validation_loss": None,
            "validation_dice": None,
        }
        assert hasattr(self.learn, "progress")

    def after_train(self):
        self.nb_batches.append(self.train_iter)

    def after_cancel_fit(self):
        self.res.update({"stop": (self.learn.epoch + 1 - 10) * self.res["validation_idxs"][0]})
        with open(self.learn.model_dir / f"{self.learn.experiment}-{self.res.get('arch')}.pkl", "wb") as f:
            pickle.dump(self.res, f)

    def after_epoch(self):
        "Plot validation loss in the pbar graph"
        if not self.nb_batches:
            return
        rec = self.learn.recorder
        iters = range_of(rec.losses)
        val_losses = [v[1] for v in rec.values]
        self.validation_dices.append(rec.log[3])

        self.res.update(
            {
                "arch": self.learn.arch.__name__,
                "batch_size": self.learn.dls.loaders[0].bs,
                "initial_learning_rate": rec.lrs[0] * 10,
                "mini_batch": iters,
                "train_loss": [float(v) for v in rec.losses],
                "validation_idxs": self.nb_batches,
                "validation_loss": val_losses,
                "validation_dice": self.validation_dices,
            }
        )

        if self.learn.n_epoch != 1 and (self.learn.epoch + 1) == (
            self.learn.n_epoch
        ):  # WARNING: Assumes all freezes are one epoch"
            self.res.update({"stop": (self.learn.epoch + 1) * self.res["validation_idxs"][0]})
            with open(self.learn.model_dir / f"{self.learn.experiment}-{self.res.get('arch')}.pkl", "wb") as f:
                pickle.dump(self.res, f)


def transfer_training(
    datasets: Dict[str, str],
    exp: str,
    mod_arch=models.resnet34,
    batch=128,
    epochs=200,
    frozen=1,
    lr=None,
    wd=None,
    save_loc: Path = Path("./models"),
    save_for_inference=True,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)  # ignore known UserWarnings about pretrained models

        dls: DataLoaders = syntrees.dataloaders(datasets.get(exp).get("path") / "train", bs=batch)
        dls.vocab = image_classes
        dls.c = len(image_classes)

        # TODO: what is the default loss function? Binary-Cross-Entropy?s
        # cbs_to_use = [CSVLogger(save_loc/f"{exp}-history.csv"), KekCallback, CBSaveModel]  # TODO -> implement callback selection!
        learn: Learner = unet_learner(
            dls,
            mod_arch,
            normalize=True,
            pretrained=True,
            metrics=Dice,
            cbs=[
                CSVLogger(save_loc / f"{exp}-history.csv"),
                KekCallback,
                EarlyStoppingCallback(monitor="train_loss", patience=10),
                SaveModelCallback(monitor="train_loss", fname=f"{exp}-{mod_arch.__name__}"),
            ],
        )  # , KekCallback, ShowGraphCallback()

        # learn.path = save_loc
        learn.model_dir = save_loc
        learn.experiment = exp
        learn.fine_tune(
            epochs, freeze_epochs=frozen, base_lr=lr or 2e-3, wd=wd
        )  # base_lr=2e-3 and wd=None are the defaults applied by fastai

        datasets.get(exp)["model_path"] = save_loc / f"{exp}.pkl"

        # learn.remove_cb(CSVLogger)

        # TODO remove, not used since I have the SaveModelCallback
        # if save_for_inference:
        #    learn.save(save_loc/f"{exp}_{learn.arch.__name__}.pkl")

        return learn


def grid_search(
    datasets: Dict[str, str],
    exp: str,
    mod_architectures=[models.resnet18, models.resnet34, models.resnet50, models.resnet101],
    batch_range=[8, 32, 128],
    lr_range=[None, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
):
    for mod_idx, mod_arch in enumerate(mod_architectures):
        for batch_idx, batch_size in enumerate(batch_range):
            for lr_idx, lr in enumerate(lr_range):
                print(f"Testing witch arch {mod_arch.__name__}, batch size of {batch_size}, base_lr of {lr}")
                _ = transfer_training(
                    datasets,
                    exp,
                    mod_arch,
                    batch=batch_size,
                    lr=lr,
                    save_loc=Path("./grid-search/models"),
                    save_for_inference=False,
                )
                del _


# Training

## Experiment 1
l1_resnet34: Learner = transfer_training(synthetic_datasets, "experiment-1")
del l1_resnet34
torch.cuda.empty_cache()
l1_resnet50: Learner = transfer_training(synthetic_datasets, "experiment-1", mod_arch=models.resnet50, batch=40)
del l1_resnet50
torch.cuda.empty_cache()

## Experiment 2
l2_resnet34: Learner = transfer_training(synthetic_datasets, "experiment-2")
del l2_resnet34
torch.cuda.empty_cache()
l2_resnet50: Learner = transfer_training(synthetic_datasets, "experiment-2", mod_arch=models.resnet50, batch=40)
del l2_resnet50
torch.cuda.empty_cache()

## Experiment 3
l3_resnet34: Learner = transfer_training(synthetic_datasets, "experiment-3")
del l3_resnet34
torch.cuda.empty_cache()
l3_resnet50: Learner = transfer_training(synthetic_datasets, "experiment-3", mod_arch=models.resnet50, batch=40)
del l3_resnet50
torch.cuda.empty_cache()

## Experiment 4
l4_resnet34: Learner = transfer_training(synthetic_datasets, "experiment-4")
del l4_resnet34
torch.cuda.empty_cache()
l4_resnet50: Learner = transfer_training(synthetic_datasets, "experiment-4", mod_arch=models.resnet50, batch=40)
del l4_resnet50
torch.cuda.empty_cache()

## Experiment 5
l5_resnet34: Learner = transfer_training(synthetic_datasets, "experiment-5")
del l5_resnet34
torch.cuda.empty_cache()
l5_resnet50: Learner = transfer_training(synthetic_datasets, "experiment-5", mod_arch=models.resnet50, batch=40)
del l5_resnet50
torch.cuda.empty_cache()
