import pandas as pd
import openpyxl
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import sys
import os
import random

from src.utils import ftir_core_dataset
from src.models import UNet, BlockViT

## Define parameters common to all experiments/models
is_local = True # used to define where data is loaded from on the HPC experiments were run on

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Data loading
test_set_fraction = 0.2
val_set_fraction= 0.2
use_augmentation = True

# Training
batch_size = 8
lr = 1e-5
l2 = 5e-1
optim_algorithm = torch.optim.Adam
max_epochs = 200

# Spectral bottleneck parameters
reduce_method = sys.argv[-3]
if reduce_method not in ['linear', 'fixed']:
    print('reduction method must be one of {linear, fixed}')
    sys.exit(1)
reduce_dim = int(sys.argv[-2]) if reduce_method == 'linear' else num_wavenumbers

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
if model_use == 'unet':
    model = UNet(num_wavenumbers, reduce_dim, n_classes, reduce_method)
elif model_use == 'blockvit':
    l2 = 5e-2
    model = BlockViT(num_wavenumbers, reduce_dim, n_classes, reduce_method,
        image_size=256,
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=192,
        num_classes=n_classes,
        mlp_dim=192*4,
    )
    optim_algorithm = torch.optim.AdamW
else:
    print("module_use must be one of {unet, blockvit}")
    sys.exit(1)
model = model.to(device)

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

dataset_train = ftir_core_dataset(
    hdf5_filepaths[where_train], mask_filepaths[where_train], annotation_filepaths[where_train],
    annotation_class_names, annotation_class_colors,
    augment=use_augmentation,
)
dataset_val = ftir_core_dataset(
    hdf5_filepaths[where_val], mask_filepaths[where_val], annotation_filepaths[where_val],
    annotation_class_names, annotation_class_colors,
    augment=False,
)
dataset_test = ftir_core_dataset(
    hdf5_filepaths[where_test], mask_filepaths[where_test], annotation_filepaths[where_test],
    annotation_class_names, annotation_class_colors,
    augment=False,
)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True,drop_last=False, generator=gen)
val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True,drop_last=False, generator=gen)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,shuffle=False,drop_last=False, generator=gen)
print(f"loader sizes:\n\ttrain: {len(train_loader)}\n\tval: {len(val_loader)}\n\ttest: {len(test_loader)}")

## Training
loss_fn = nn.CrossEntropyLoss(reduction='none')
optimizer = optim_algorithm(model.parameters(), lr=lr,weight_decay=l2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=15, threshold=0.01, cooldown=0)

training_losses,validation_losses = [],[]
training_accs,validation_accs = [],[]
training_f1ms,validation_f1ms = [],[]
training_f1s,validation_f1s = [],[]
lr_decreases = []
current_iters = 0
best_val_f1 = 0
best_val_iter = 0
stop_training=False

for epoch in range(max_epochs):
    print(f"\n ON EPOCH {epoch + 1}")

    # reset running metrics
    running_loss_train, running_loss_val = 0, 0
    train_preds, train_targets = [], []
    val_preds, val_targets = [], []

    # Train loop
    model.train()
    for batch_idx, (data, annot, mask, has_annotations) in enumerate(train_loader):

        # Put data and label on device
        data = data.to(device);
        annot = annot.to(device);
        has_annotations = has_annotations.to(device)

        # Push data through model
        optimizer.zero_grad()
        out = model(data)

        # Calculate loss
        loss = loss_fn(out, annot.argmax(dim=1)) * has_annotations  # loss per pixel with annotations
        loss = loss.sum() / (has_annotations.sum())  # mean loss per annotated pixel
        loss.backward()  # backprop
        optimizer.step()

        # Calculate metrics
        running_loss_train += loss.cpu().item()
        targets = annot.argmax(dim=1)[has_annotations]  # class targets on annotated pixels
        preds = out.argmax(dim=1)[has_annotations]  # predicted values on annotated pixels
        train_preds.extend(preds.detach().cpu().numpy())
        train_targets.extend(targets.detach().cpu().numpy())

    # Validate loop
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, annot, mask, has_annotations) in enumerate(val_loader):

            # Put data and label on device
            data = data.to(device);
            annot = annot.to(device);
            has_annotations = has_annotations.to(device)

            # Push data through model
            out = model(data)

            # Calculate loss
            loss = loss_fn(out, annot.argmax(dim=1)) * has_annotations  # loss per pixel
            loss = loss.sum() / (has_annotations.sum())  # mean loss per annotated pixel

            # Calculate metrics
            running_loss_val += loss.cpu().item()
            targets = annot.argmax(dim=1)[has_annotations]  # class targets on annotated pixels
            preds = out.argmax(dim=1)[has_annotations]  # predicted values on annotated pixels
            val_preds.extend(preds.detach().cpu().numpy())
            val_targets.extend(targets.detach().cpu().numpy())

    # calculate epoch metrics for train set
    train_acc = accuracy_score(train_targets, train_preds);
    training_accs.append(train_acc)
    train_f1m = f1_score(train_targets, train_preds, average='macro');
    training_f1ms.append(train_f1m)
    train_f1 = f1_score(train_targets, train_preds, average=None);
    training_f1s.append(train_f1)
    train_loss = running_loss_train / (len(dataset_train));
    training_losses.append(train_loss)

    # calculate epoch metrics for val set
    val_acc = accuracy_score(val_targets, val_preds);
    validation_accs.append(val_acc)
    val_f1m = f1_score(val_targets, val_preds, average='macro');
    validation_f1ms.append(val_f1m)
    val_f1 = f1_score(val_targets, val_preds, average=None);
    validation_f1s.append(val_f1)
    val_loss = running_loss_val / (len(dataset_val));
    validation_losses.append(val_loss)

    # Update
    print(f"TRAIN --- | Loss: {train_loss:.4} | OA: {train_acc:.4} | f1: {train_f1m:.4}")
    print(f"VAL ----- | Loss: {val_loss:.4} | OA: {val_acc:.4} | f1: {val_f1m:.4}")

    # If performance on validation set best so far, save model
    if val_f1m > best_val_f1:
        best_val_f1 = val_f1m
        best_val_epoch = epoch
        if not is_local:
            torch.save(model.state_dict(), rf'./{model_use}_{reduce_method}_weights_{seed}.pt')

    # Step the scheduler based on the validation set performance
    scheduler.step(val_f1m)
    new_lr = optimizer.param_groups[0]['lr']
    if new_lr != lr:
        print(f"Val f1 plateaued, lr {lr} -> {new_lr}")
        lr = new_lr
        lr_decreases.append(epoch)
        if len(lr_decreases) >= 3:
            print("Val f1 decreased thrice, ending training early")
            break


if not is_local:
    model.load_state_dict(torch.load(rf'./{model_use}_{reduce_method}_weights_{seed}.pt', weights_only=True))

# Test
running_loss_test = 0
test_preds, test_targets = [], []
model.eval()
with torch.no_grad():
    for batch_idx, (data, annot, mask, has_annotations) in enumerate(test_loader):
        # Put data and label on device
        data = data.to(device);
        annot = annot.to(device);
        has_annotations = has_annotations.to(device)

        # Push data through model
        out = model(data)

        # Calculate loss
        loss = loss_fn(out, annot.argmax(dim=1)) * has_annotations  # loss per pixel
        loss = loss.sum() / (has_annotations.sum())  # mean loss per annotated pixel

        # Calculate metrics
        running_loss_test += loss.cpu().item()
        targets = annot.argmax(dim=1)[has_annotations]  # class targets on annotated pixels
        preds = out.argmax(dim=1)[has_annotations]  # predicted values on annotated pixels
        test_preds.extend(preds.detach().cpu().numpy())
        test_targets.extend(targets.detach().cpu().numpy())

# calculate test set metrics
test_acc = accuracy_score(test_targets, test_preds)
test_f1m = f1_score(test_targets, test_preds, average='macro')
test_f1 = f1_score(test_targets, test_preds, average=None)
test_loss = running_loss_test / batch_idx

print(f"TEST ---- | Loss: {test_loss:.4} | OA: {test_acc:.4} | f1: {test_f1m:.4}")
for cls_idx, f1 in enumerate(test_f1):
    print(f"{annotation_class_names[cls_idx]}{(20 - len(annotation_class_names[cls_idx])) * ' '} : {f1:.4}")



## Evaluation

# Plot overall performance curves per-epoch
fig,ax = plt.subplots(1,3,figsize=(16,5))
ax[0].plot(np.arange(1,len(training_losses)+1),np.array(training_losses),color='cornflowerblue',label="train")
ax[0].plot(np.arange(1,len(validation_losses)+1),np.array(validation_losses),color='orange',label="validation")
ax[0].scatter(len(validation_losses),test_loss,color='green',label="test",marker="x")
ax[0].set_title("loss curves"); ax[0].legend()

ax[1].plot(np.arange(1,len(training_accs)+1),np.array(training_accs),color='cornflowerblue',label="train")
ax[1].plot(np.arange(1,len(validation_accs)+1),np.array(validation_accs),color='orange',label="validation")
ax[1].scatter(len(validation_losses),test_acc,color='green',label="test",marker="x")
ax[1].set_title("accuracy"); ax[1].legend()

ax[2].plot(np.arange(1,len(training_f1ms)+1),np.array(training_f1ms),color='cornflowerblue',label="train")
ax[2].plot(np.arange(1,len(validation_f1ms)+1),np.array(validation_f1ms),color='orange',label="validation")
ax[2].scatter(len(validation_losses),test_f1m,color='green',label="test",marker="x")
ax[2].set_title("macro f1"); ax[2].legend()

for lrd in lr_decreases:
    ax[0].axvline(x=lrd, ymin=0, ymax=1, color='grey')
    ax[1].axvline(x=lrd, ymin=0, ymax=1, color='grey')
    ax[2].axvline(x=lrd, ymin=0, ymax=1, color='grey')

ax[0].axvline(x=best_val_epoch, ymin=0, ymax=1, color='red',alpha=0.3)
ax[1].axvline(x=best_val_epoch, ymin=0, ymax=1, color='red',alpha=0.3)
ax[2].axvline(x=best_val_epoch, ymin=0, ymax=1, color='red',alpha=0.3)

plt.tight_layout()
if not is_local:
    plt.savefig(f'./loss_curve_{model_use}_{reduce_method}_{seed}.png')
    plt.close(fig)

# Plot per-class f1 score curves
fig,ax = plt.subplots(2,3,figsize=(15,5)); ax = ax.flatten()
for cls in range(6):
    ax[cls].plot(np.arange(1,len(training_f1s)+1),[i[cls] for i in training_f1s], color='black', label="train")
    ax[cls].plot(np.arange(1,len(validation_f1s)+1),[i[cls] for i in validation_f1s], color=annotation_class_colors[cls]/255, label="val")
    ax[cls].set_title(f"{annotation_class_names[cls]}")
    ax[cls].legend()
    ax[cls].scatter(len(validation_losses),test_f1[cls],color='green',label="test",marker="x")
    for lrd in lr_decreases:
        ax[cls].axvline(x=lrd, ymin=0, ymax=1, color='grey')
    ax[cls].axvline(x=best_val_epoch, ymin=0, ymax=1, color='red',alpha=0.3)
fig.suptitle("Class-specific F1 scores")
plt.tight_layout()
if not is_local:
    plt.savefig(f'./loss_curve_byclass_{model_use}_{reduce_method}_{seed}.png')
    plt.close(fig)

## Save model and results

# Save model
if not is_local:
    model = model.cpu()
    torch.save(model.state_dict(), rf'./{model_use}_{reduce_method}_weights_{seed}.pt')

# save results
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
