# spatial-spectral-prostate

Training scripts and model implementations for the paper "Spatial-Spectral Deep Learning for Prostate Cancer Tissue Classification in Infrared Spectroscopy".

## Training scripts

Training scripts are run in the format `script.py model bottleneck bottleneck_size random_seed` 

Examples:  
25px window Patch-CNN with no bottleneck: `patch_based_training.py patch_25px fixed 0 42`  
Block-ViT model with K=64 linear bottleneck: `core_based_training.py blockvit linear 64 42`  
RandomForest model with K=16 PCA bottleneck: `classical_model_training.py randomforest pca 16 42`

## Prediction scripts

We include sample prediction scripts as notebooks `classical_model_prediction.ipynb`, `patch_based_prediction.ipynb`, and `core_based_prediction.ipynb`. 
Please note that patch-based prediction can be very slow for some of the larger patch sizes.

## Annotation colours
Please note that annotation class colours are different in these scripts compared to the paper. The colour scheme was revised to aid legibility.
A key to translate colours is as follows (RGB):  

Here:
- 0: normal epithelium = limegreen = `[0, 255, 0]`
- 1: normal stroma = purple = `[128,0,128]`
- 2: cancerous epithelium = fuchsia = `[255,0,255]`
- 3: cancer-associated stroma = blue = `[0,0,255]`
- 4: corpora amylacea = orange = `[255,165,0]`
- 5: blood = red = `[255,0,0]`

Paper:
- 0: normal epithelium = cornflower blue = `[51, 204, 255]`
- 1: normal stroma = brown = `[153, 102, 51]`
- 2: cancerous epithelium = magenta = `[255, 0, 102]`
- 3: cancer-associated stroma = tan = `[255, 190, 142]`
- 4: corpora amylacea = limegreen = `[0, 255, 0]`
- 5: blood = yellow = `[255, 255, 0]`
