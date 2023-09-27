#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# # Setup

# In[2]:


synthetic_datasets = [(s.split("/")[-1].replace("_", "-"), s) for s in glob.glob("./data/experiment_[1-8]")]
synthetic_datasets = {k: {"path": Path(v)} for k, v in synthetic_datasets if glob.glob(v + "/**/*")}


# In[3]:


image_classes: List[str] = ['other', 'tree']


# In[4]:


def get_these_ys(p: Path):
    id, *split_fname = p.stem.split("_")
    f = p.parent.parent/"valid"/f"{id}_gt_{'_'.join(split_fname)}{p.suffix}"
    msk = np.array(Image.open(f))
    return PILMask.create(msk)


# In[5]:


syntrees = DataBlock(
    blocks=(ImageBlock, MaskBlock(image_classes)),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.3),
    get_y=get_these_ys,
    item_tfms=RandomResizedCrop((256,256)),
    batch_tfms=[Dihedral(), Brightness(max_lighting=0.4), Contrast(), Saturation(0.8)]
)


# In[6]:


syntrees_inf = DataBlock(
    blocks=(ImageBlock, MaskBlock(image_classes)),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.3),
    get_y=get_these_ys,
    #item_tfms=RandomResizedCrop((256,256)),
    batch_tfms=[Dihedral(), Brightness(max_lighting=0.4), Contrast(), Saturation(0.8)]
)


# In[7]:


class KekCallback(Callback):
    "Shamelessly stolen from ShowGraphCallback"
    order,run_valid=65,False

    def before_fit(self):
        self.run = not hasattr(self.learn, 'lr_finder') and not hasattr(self, "gather_preds")
        if not(self.run): return
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
            "validation_dice": None
        }
        assert hasattr(self.learn, 'progress')

    def after_train(self): self.nb_batches.append(self.train_iter)

    def after_cancel_fit(self):
        self.res.update({"stop": (self.learn.epoch + 1 - 10) * self.res["validation_idxs"][0]})
        with open(self.learn.model_dir/f"{self.learn.experiment}-{self.res.get('arch')}.pkl", "wb") as f:
            pickle.dump(self.res, f)

    def after_epoch(self):
        "Plot validation loss in the pbar graph"
        if not self.nb_batches: return
        rec = self.learn.recorder        
        iters = range_of(rec.losses)
        val_losses = [v[1] for v in rec.values]
        self.validation_dices.append(rec.log[3])
        
        self.res.update({"arch": self.learn.arch.__name__,
        "batch_size": self.learn.dls.loaders[0].bs,
        "initial_learning_rate": rec.lrs[0] * 10,
        "mini_batch": iters,
        "train_loss": [float(v) for v in rec.losses],
        "validation_idxs": self.nb_batches,
        "validation_loss": val_losses,
        "validation_dice": self.validation_dices})
	
        if self.learn.n_epoch != 1 and (self.learn.epoch + 1) == (self.learn.n_epoch):  # WARNING: Assumes all freezes are one epoch"            
            self.res.update({"stop": (self.learn.epoch + 1) * self.res["validation_idxs"][0]})
            with open(self.learn.model_dir/f"{self.learn.experiment}-{self.res.get('arch')}.pkl", "wb") as f:
                pickle.dump(self.res, f)


# In[15]:


def transfer_training(datasets: Dict[str, str], exp: str, mod_arch=models.resnet34, batch=128, epochs=200, frozen=1, lr=None, wd=None, save_loc: Path=Path("./models"), save_for_inference=True) -> None:
     with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning) # ignore known UserWarnings about pretrained models
        
        dls: DataLoaders = syntrees.dataloaders(datasets.get(exp).get("path")/"train", bs=batch)
        dls.vocab = image_classes
        dls.c = len(image_classes)
        
        # TODO: what is the default loss function? Binary-Cross-Entropy?s
        #cbs_to_use = [CSVLogger(save_loc/f"{exp}-history.csv"), KekCallback, CBSaveModel]  # TODO -> implement callback selection!
        learn: Learner = unet_learner(dls, mod_arch, normalize=True, pretrained=True, metrics=Dice, cbs=[CSVLogger(save_loc/f"{exp}-history.csv"), KekCallback, EarlyStoppingCallback(monitor="train_loss", patience=10), SaveModelCallback(monitor="train_loss", fname=f"{exp}-{mod_arch.__name__}")]) # , KekCallback, ShowGraphCallback()
        
        #learn.path = save_loc
        learn.model_dir = save_loc
        learn.experiment = exp
        learn.fine_tune(epochs, freeze_epochs=frozen, base_lr=lr or 2e-3, wd=wd)  # base_lr=2e-3 and wd=None are the defaults applied by fastai
        
        datasets.get(exp)["model_path"] = save_loc/f"{exp}.pkl"
        
        #learn.remove_cb(CSVLogger)        
        
        # TODO remove, not used since I have the SaveModelCallback
        #if save_for_inference:    
        #    learn.save(save_loc/f"{exp}_{learn.arch.__name__}.pkl")
        
        return learn


# In[9]:


def dice(p: torch.tensor, t: torch.tensor) -> float:
    tp = np.sum(np.array(p) * np.array(t))
    area = np.sum(np.array(p)) + np.sum(np.array(t)) # 2 * tp + fn + fp => (tp + fp) + (tp + fn) => p == 1 + t == 1
    if  tp == 0 and area == 0:
        return 1
    else:
        return float((2 * tp) / area)


# In[10]:


def NicerDicer(truths: List[torch.tensor], predictions: List[torch.tensor]) -> float:
    assert(len(truths) == len(predictions))
    added_dices = []
    for pred, truth in zip(predictions, truths):
        added_dices.append(float(dice(pred, truth)))
    return sum(added_dices) / len(truths)


# In[11]:


# https://benjaminwarner.dev/2021/10/01/inference-with-fastai
# https://medium.com/mlearning-ai/understanding-evaluation-metrics-in-medical-image-segmentation-d289a373a3f
def inference(ds: Path, datasets: Optional[Dict[str, str]]=None, exp: Optional[str]=None, long_path: Optional[Path]=None, mod_arch=models.resnet34) -> Tuple[List[TensorBase], float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning) # ignore known UserWarnings about pretrained models
        dls: DataLoaders = syntrees_inf.dataloaders(ds/"real", bs=8) # pseudo data set; not used; batch size could be anything but acutally does have an influence
        dls.vocab = image_classes
        dls.c = len(image_classes)
        
        inference_learner = unet_learner(dls, mod_arch, normalize=True, pretrained=True, metrics=Dice)
        inference_learner.load(datasets.get(exp).get("model_path") if datasets and exp else long_path, device='cuda')
        
        input_paths: L = get_image_files(ds, folders="real")
        dls = inference_learner.dls.test_dl(input_paths, with_decoded=True)
        mdls = [cast(np.array(get_these_ys(f), np.float64), TensorBase) for f in input_paths] # ehmm... get validation masks and convert to tensors
        
        preds, _ = inference_learner.get_preds(dl = dls)
        flattened_preds = [preds[i].argmax(dim=0) for i in range(0, preds.shape[0])]
    
        test_avg_dice = NicerDicer(mdls, flattened_preds)
        
        return flattened_preds, test_avg_dice, input_paths


# In[12]:


def grid_search(datasets: Dict[str, str], exp: str, 
                mod_architectures=[models.resnet18, models.resnet34, models.resnet50, models.resnet101], 
                batch_range=[8, 32, 128], 
                lr_range=[None, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]):
    for mod_idx, mod_arch in enumerate(mod_architectures):
        for batch_idx, batch_size in enumerate(batch_range):
            for lr_idx, lr in enumerate(lr_range):
                print(f"Testing witch arch {mod_arch.__name__}, batch size of {batch_size}, base_lr of {lr}")
                _ = transfer_training(datasets, exp, mod_arch, batch=batch_size, lr=lr, save_loc=Path("./grid-search/models"), save_for_inference=False)
                del _


# In[13]:


def array2img(arr: TensorBase, out_path: Path, orig: Path) -> None:
    im: Image = Image.fromarray(np.array(arr, dtype=np.uint8) * np.iinfo(np.uint8).max)
    im.save(out_path / orig.name)


# # Grid Search

# ## Experiment 1
# 
# **Das habe ich dann doch aufgegeben!**

# In[14]:


#grid_search(synthetic_datasets, "experiment-1")


# # Training

# ## Experiment 1

# In[18]:


torch.cuda.empty_cache()


# In[14]:


l1_resnet34: Learner = transfer_training(synthetic_datasets, "experiment-1")
del l1_resnet34
torch.cuda.empty_cache()
l1_resnet50: Learner = transfer_training(synthetic_datasets, "experiment-1", mod_arch=models.resnet50, batch=40)
del l1_resnet50
torch.cuda.empty_cache()


# ## Experiment 2

# In[ ]:


l2_resnet34: Learner = transfer_training(synthetic_datasets, "experiment-2")
del l2_resnet34
torch.cuda.empty_cache()
l2_resnet50: Learner = transfer_training(synthetic_datasets, "experiment-2", mod_arch=models.resnet50, batch=40)
del l2_resnet50
torch.cuda.empty_cache()


# ## Experiment 3

# In[14]:


l3_resnet34: Learner = transfer_training(synthetic_datasets, "experiment-3")
del l3_resnet34
torch.cuda.empty_cache()
l3_resnet50: Learner = transfer_training(synthetic_datasets, "experiment-3", mod_arch=models.resnet50, batch=40)
del l3_resnet50
torch.cuda.empty_cache()


# ## Experiment 4

# In[14]:


l4_resnet34: Learner = transfer_training(synthetic_datasets, "experiment-4")
del l4_resnet34
torch.cuda.empty_cache()
l4_resnet50: Learner = transfer_training(synthetic_datasets, "experiment-4", mod_arch=models.resnet50, batch=40)
del l4_resnet50
torch.cuda.empty_cache()


# ## Experiment 8

# In[ ]:
l5_resnet34: Learner = transfer_training(synthetic_datasets, "experiment-5")
del l5_resnet34
torch.cuda.empty_cache()
l5_resnet50: Learner = transfer_training(synthetic_datasets, "experiment-5", mod_arch=models.resnet50, batch=40)
del l5_resnet50
torch.cuda.empty_cache()


# # Testing/ Inference

# ## Experiment 1


flat_preds_1_resnet34, d_1_resnet34, test_paths_1_resnet34 = inference(long_path=Path("./experiment-1-resnet34"), ds=Path("./data/validation"), mod_arch=models.resnet34)
print(f"Model 1 (ResNet34) achieved a dice accuracy of {round(d_1_resnet34, 4)}")

for a, f in zip(flat_preds_1_resnet34, test_paths_1_resnet34):
    array2img(a, Path("./data/validation/prediction-masks/experiment-1_resnet34/"), f)

flat_preds_1_resnet50, d_1_resnet50, test_paths_1_resnet50 = inference(long_path=Path("./experiment-1-resnet50"), ds=Path("./data/validation"), mod_arch=models.resnet50)
print(f"Model 1 (ResNet50) achieved a dice accuracy of {round(d_1_resnet50, 4)}")

for a, f in zip(flat_preds_1_resnet50, test_paths_1_resnet50):
    array2img(a, Path("./data/validation/prediction-masks/experiment-1_resnet50/"), f)


# ## Experiment 2

flat_preds_2_resnet34, d_2_resnet34, test_paths_2_resnet34 = inference(long_path=Path("./experiment-2-resnet34"), ds=Path("./data/validation"), mod_arch=models.resnet34)
print(f"Model 2 (ResNet34) achieved a dice accuracy of {round(d_2_resnet34, 4)}")

for a, f in zip(flat_preds_2_resnet34, test_paths_2_resnet34):
    array2img(a, Path("./data/validation/prediction-masks/experiment-2_resnet34/"), f)

flat_preds_2_resnet50, d_2_resnet50, test_paths_2_resnet50 = inference(long_path=Path("./experiment-2-resnet50"), ds=Path("./data/validation"), mod_arch=models.resnet50)
print(f"Model 2 (ResNet50) achieved a dice accuracy of {round(d_2_resnet50, 4)}")

for a, f in zip(flat_preds_2_resnet50, test_paths_2_resnet50):
    array2img(a, Path("./data/validation/prediction-masks/experiment-2_resnet50/"), f)


# ## Experiment 3

flat_preds_3_resnet34, d_3_resnet34, test_paths_3_resnet34 = inference(long_path=Path("./experiment-3-resnet34"), ds=Path("./data/validation"), mod_arch=models.resnet34)
print(f"Model 3 (ResNet34) achieved a dice accuracy of {round(d_3_resnet34, 4)}")

for a, f in zip(flat_preds_3_resnet34, test_paths_3_resnet34):
    array2img(a, Path("./data/validation/prediction-masks/experiment-3_resnet34/"), f)

flat_preds_3_resnet50, d_3_resnet50, test_paths_3_resnet50 = inference(long_path=Path("./experiment-3-resnet50"), ds=Path("./data/validation"), mod_arch=models.resnet50)
print(f"Model 3 (ResNet50) achieved a dice accuracy of {round(d_3_resnet50, 4)}")

for a, f in zip(flat_preds_3_resnet50, test_paths_3_resnet50):
    array2img(a, Path("./data/validation/prediction-masks/experiment-3_resnet50/"), f)


# ## Experiment 4

flat_preds_4_resnet34, d_4_resnet34, test_paths_4_resnet34 = inference(long_path=Path("./experiment-4-resnet34"), ds=Path("./data/validation"), mod_arch=models.resnet34)
print(f"Model 4 (ResNet34) achieved a dice accuracy of {round(d_4_resnet34, 4)}")

for a, f in zip(flat_preds_4_resnet34, test_paths_4_resnet34):
    array2img(a, Path("./data/validation/prediction-masks/experiment-4_resnet34/"), f)

flat_preds_4_resnet50, d_4_resnet50, test_paths_4_resnet50 = inference(long_path=Path("./experiment-4-resnet50"), ds=Path("./data/validation"), mod_arch=models.resnet50)
print(f"Model 4 (ResNet50) achieved a dice accuracy of {round(d_4_resnet50, 4)}")

for a, f in zip(flat_preds_4_resnet50, test_paths_4_resnet50):
    array2img(a, Path("./data/validation/prediction-masks/experiment-4_resnet50/"), f)


# ## Experiment 5

flat_preds_5_resnet34, d_5_resnet34, test_paths_5_resnet34 = inference(long_path=Path("./models/experiment-5_resnet34.pkl"), ds=Path("./data/validation"), mod_arch=models.resnet34)
print(f"Model 5 (ResNet34) achieved a dice accuracy of {round(d_5_resnet34, 4)}")

for a, f in zip(flat_preds_5_resnet34, test_paths_5_resnet34):
    array2img(a, Path("./data/validation/prediction-masks/experiment-5_resnet34/"), f)

flat_preds_5_resnet50, d_5_resnet50, test_paths_5_resnet50 = inference(long_path=Path("./models/experiment-5_resnet50.pkl"), ds=Path("./data/validation"), mod_arch=models.resnet50)
print(f"Model 5 (ResNet50) achieved a dice accuracy of {round(d_5_resnet50, 4)}")

for a, f in zip(flat_preds_5_resnet50, test_paths_5_resnet50):
    array2img(a, Path("./data/validation/prediction-masks/experiment-5_resnet50/"), f)

