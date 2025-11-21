import pandas as pd
import openpyxl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch
from sklearn.decomposition import PCA
import sys
import os
import random
import pickle

from src.models import SVM, RandomForest
from src.utils import ftir_patching_dataset

## Define parameters common to all experiments/models
is_local = True # todo

# Experiment
seed = int(sys.argv[-1])
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
gen = torch.Generator()
gen.manual_seed(seed)
image_size = 256
num_wavenumbers = 965

# Data loading
test_set_fraction = 0.2
val_set_fraction= 0.2

# Training
samples_to_train = 1000 #todo

# Spectral bottleneck parameters
reduce_method = sys.argv[-3]
if reduce_method not in ['pca', 'fixed']:
    print('reduction method must be one of {pca, fixed}')
    sys.exit(1)
reduce_dim = int(sys.argv[-2]) if reduce_method == 'pca' else num_wavenumbers

## Import dataset
def csf_fp(filepath):
    return filepath.replace('D:/datasets','D:/datasets' if is_local else '../')
master = pd.read_excel(csf_fp(rf'D:/datasets/pcuk2023_ftir_whole_core/master_sheet.xlsx'))
slide = master['slide'].to_numpy()
patient_id = master['patient_id'].to_numpy()
hdf5_filepaths = np.array([csf_fp(fp) for fp in master['hdf5_filepath']])
annotation_filepaths = np.array([csf_fp(fp) for fp in master['annotation_filepath']])
mask_filepaths = np.array([csf_fp(fp) for fp in master['mask_filepath']])
wavenumbers = np.load(csf_fp(f'D:/datasets/pcuk2023_ftir_whole_core/wavenumbers.npy'))
annotation_class_colors = np.array([[0,255,0],[128,0,128],[255,0,255],[0,0,255],[255,165,0],[255,0,0]])
annotation_class_names = np.array(['epithelium_n','stroma_n','epithelium_c','stroma_c','corpora_amylacea','blood'])
n_classes = len(annotation_class_names)

print(f"Loaded {len(slide)} cores, with {n_classes} classes, and {len(wavenumbers)} wavenumbers")

## Import experiment-specific parameters
model_use = sys.argv[-4]
if model_use == 'randomforest':
    model = RandomForest()
elif model_use == 'svm':
    model = SVM()
else:
    print("module_use must be one of {randomforest, svm}")
    sys.exit(1)
if reduce_method == 'pca':
    model.steps.insert(0,['pca', PCA(n_components=reduce_dim)])

## Split data and define loaders
unique_pids = np.unique(patient_id)
pids_trainval, pids_test, _, _ = train_test_split(
    unique_pids, np.zeros_like(unique_pids), test_size=test_set_fraction, random_state=seed)
pids_train, pids_val, _, _ = train_test_split(
    pids_trainval, np.zeros_like(pids_trainval), test_size=(val_set_fraction/(1-test_set_fraction)), random_state=seed)
where_train = np.where(np.isin(patient_id,pids_train))
where_val = np.where(np.isin(patient_id,pids_val))
where_test = np.where(np.isin(patient_id,pids_test))
print(f"Cores per data split:\n\tTRAIN: {len(where_train[0])}\n\tVAL: {len(where_val[0])}\n\tTEST: {len(where_test[0])}")

dataset_train = ftir_patching_dataset(
    hdf5_filepaths[where_train], mask_filepaths[where_train], annotation_filepaths[where_train],
    annotation_class_names, annotation_class_colors,
    image_size=image_size, patch_dim = 1, augment=False,
)
dataset_val = ftir_patching_dataset(
    hdf5_filepaths[where_val], mask_filepaths[where_val], annotation_filepaths[where_val],
    annotation_class_names, annotation_class_colors,
    image_size=image_size, patch_dim = 1, augment=False,
)
dataset_test = ftir_patching_dataset(
    hdf5_filepaths[where_test], mask_filepaths[where_test], annotation_filepaths[where_test],
    annotation_class_names, annotation_class_colors,
    image_size=image_size, patch_dim = 1, augment=False,
)

# Instantiate data loaders
# Train + Val dsets weighted for equal class distribution
_, class_counts = np.unique(dataset_train.tissue_classes, return_counts=True)
class_weights = 1 / class_counts
class_weights = class_weights[dataset_train.tissue_classes]
train_sampler = torch.utils.data.WeightedRandomSampler(class_weights, len(class_weights), replacement=True)

_, class_counts = np.unique(dataset_val.tissue_classes, return_counts=True)
class_weights = 1 / class_counts
class_weights = class_weights[dataset_val.tissue_classes]
val_sampler = torch.utils.data.WeightedRandomSampler(class_weights, len(class_weights), replacement=True)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=1000, sampler=train_sampler,drop_last=False, generator=gen)
val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=1000, sampler=val_sampler,drop_last=False, generator=gen)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1000,shuffle=False,drop_last=False, generator=gen)
print(f"loader sizes:\n\ttrain: {len(train_loader)}\n\tval: {len(val_loader)}\n\ttest: {len(test_loader)}")

## Sample data for training
train_data = []
train_labels = []
for iter in range(0, (samples_to_train // 1000) + 1):
    for bidx, (data, label) in enumerate(train_loader):
        train_data.append(data.squeeze().numpy())
        train_labels.append(label.squeeze().numpy())

        if iter * len(train_loader) + bidx * 1000 > samples_to_train:
            end = True
            break
    if end:
        break
train_data = np.concatenate(train_data, axis=0)[:samples_to_train]
train_labels = np.concatenate(train_labels, axis=0)[:samples_to_train]

## Fit model to train data
model.fit(train_data,train_labels)
print(f"Model accuracy on the train data: {accuracy_score(model.predict(train_data),train_labels)}")

## Evaluate model on test data
test_preds, test_targets = [], []

for bidx, (data, label) in enumerate(test_loader):
    print(f"{bidx}/{(len(test_loader))}", end="\r")
    data = data.squeeze().numpy()
    label = label.squeeze().numpy()
    pred= model.predict(data)

    test_preds.extend(pred)
    test_targets.extend(label)

test_targets = np.array(test_targets);
test_preds = np.array(test_preds)
test_acc = accuracy_score(test_targets, test_preds)
test_f1m = f1_score(test_targets, test_preds, average='macro')
test_f1 = f1_score(test_targets, test_preds, average=None)
print("Metrics on entire testing set:")
print(f"TEST ---- | OA: {test_acc:.4f} | f1: {test_f1m:.4f}")
for cls_idx, f1 in enumerate(test_f1):
    print(f"{annotation_class_names[cls_idx]}{(20 - len(annotation_class_names[cls_idx])) * ' '} : {f1:.4f}")

## Save model and results

# Save model
if not is_local:
    with open(f'./{model_use}_{reduce_method}_weights_{seed}.pt','wb') as f:
        pickle.dump(model,f)

# Save results
if not is_local:
    if os.path.isfile(f'results_{model_use}_{reduce_method}.txt'):
        f = open(f'results_{model_use}_{reduce_method}.txt', 'r')
        lines = f.readlines()
        f.close()
    else:
        lines = [x + ', \n' for x in ['seed', *annotation_class_names, 'overall_acc', 'macro_f1']]

    # Process files
    lines[0] = lines[0].replace('\n', str(seed) + ', \n')
    for cls in range(n_classes):
        lines[cls + 1] = lines[cls + 1].replace('\n', str(test_f1[cls]) + ', \n')
    lines[n_classes + 1] = lines[n_classes + 1].replace('\n', str(test_acc) + ', \n')
    lines[n_classes + 2] = lines[n_classes + 2].replace('\n', str(test_f1m) + ', \n')

    f = open(f'results_{model_use}_{reduce_method}.txt', 'w')
    f.write(''.join(lines))
    f.close()
