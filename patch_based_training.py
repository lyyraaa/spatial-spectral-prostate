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

from src.utils import ftir_patching_dataset
from src.models import MLP, patch3_cnn, patch25_cnn, patch101_cnn, patch_multiscale, patch25_transformer

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Data loading
test_set_fraction = 0.2
val_set_fraction= 0.2
use_augmentation = True

# Network parameters
dropout_p = 0.5

# Training
batch_size = 64
lr = 1e-5
l2 = 5e-1
optim_algorithm = torch.optim.Adam
max_iters = 5000
pseudo_epoch = 100

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
if model_use == 'mlp':
    model = MLP(num_wavenumbers, reduce_dim, n_classes, reduce_method, dropout_p=dropout_p)
    patch_size = 1
elif model_use == 'patch_3px':
    model = patch3_cnn(num_wavenumbers, reduce_dim, n_classes, reduce_method, dropout_p=dropout_p)
    patch_size = 3
elif model_use == 'patch_25px':
    model = patch25_cnn(num_wavenumbers, reduce_dim, n_classes, reduce_method, dropout_p=dropout_p)
    patch_size = 25
elif model_use == 'patch_101px':
    model = patch101_cnn(num_wavenumbers, reduce_dim, n_classes, reduce_method, dropout_p=dropout_p)
    patch_size = 101
elif model_use == 'patch_multiscale':
    model = patch_multiscale(num_wavenumbers, reduce_dim, n_classes, reduce_method, dropout_p=dropout_p)
    patch_size = 101
elif model_use == 'patch_transformer':
    lr = 5e-4
    l2 = 5e-2
    patch_size = 25
    optim_algorithm = torch.optim.AdamW
    model = patch25_transformer(num_wavenumbers, reduce_dim, n_classes, 256, reduce_method)
else:
    print("module_use must be one of {mlp, patch_3px, patch_25px, patch_101px, patch_multiscale, patch_transformer}")
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

dataset_train = ftir_patching_dataset(
    hdf5_filepaths[where_train], mask_filepaths[where_train], annotation_filepaths[where_train],
    annotation_class_names, annotation_class_colors,
    image_size=image_size, patch_dim = patch_size, augment=use_augmentation,
)
dataset_val = ftir_patching_dataset(
    hdf5_filepaths[where_val], mask_filepaths[where_val], annotation_filepaths[where_val],
    annotation_class_names, annotation_class_colors,
    image_size=image_size, patch_dim = patch_size, augment=False,
)
dataset_test = ftir_patching_dataset(
    hdf5_filepaths[where_test], mask_filepaths[where_test], annotation_filepaths[where_test],
    annotation_class_names, annotation_class_colors,
    image_size=image_size, patch_dim = patch_size, augment=False,
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

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, sampler=train_sampler,drop_last=False, generator=gen)
val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, sampler=val_sampler,drop_last=False, generator=gen)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,shuffle=True,drop_last=False, generator=gen)
print(f"loader sizes:\n\ttrain: {len(train_loader)}\n\tval: {len(val_loader)}\n\ttest: {len(test_loader)}")


## Training
loss_fn = nn.CrossEntropyLoss()
optimizer = optim_algorithm(model.parameters(), lr=lr,weight_decay=l2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=500, threshold=0.01, cooldown=250)

training_losses,validation_losses = [],[]
training_accs,validation_accs = [],[]
training_f1ms,validation_f1ms = [],[]
training_f1s,validation_f1s = [],[]
lr_decreases = []
current_iters = 0
best_val_f1 = 0
best_val_iter = 0
stop_training=False

# training loop
while current_iters < max_iters:
    for (bidx, (data, label)) in enumerate(train_loader):
        data = data.to(device);
        label = label.to(device)

        # Push through model
        model.train()
        optimizer.zero_grad()
        out = model(data)

        # Calculate loss
        loss = loss_fn(out, label)
        loss.backward()
        optimizer.step()

        # Append log arrays
        training_losses.append(loss.item())
        pred = out.argmax(dim=1).detach().cpu().numpy()
        actual = label.cpu().numpy()
        training_accs.append(accuracy_score(actual, pred))
        training_f1ms.append(f1_score(actual, pred, average='macro'))
        training_f1s.append(f1_score(actual, pred, average=None, labels=np.arange(0, n_classes), zero_division=0))

        # Do validation cycle
        model.eval()
        with torch.no_grad():
            # load data
            data, label = next(iter(val_loader))
            data = data.to(device);
            label = label.to(device)

            # Push through model
            out = model(data)

            # Calculate loss
            loss = loss_fn(out, label)

            # Append log arrays
            validation_losses.append(loss.item())
            pred = out.argmax(dim=1).detach().cpu().numpy()
            actual = label.cpu().numpy()
            validation_accs.append(accuracy_score(actual, pred))
            validation_f1ms.append(f1_score(actual, pred, average='macro'))
            validation_f1s.append(f1_score(actual, pred, average=None, labels=np.arange(0, n_classes), zero_division=0))

        # Print training statistics every N iters
        if current_iters % pseudo_epoch == 0:
            print(f"ON ITER: {current_iters}, metrics for last {pseudo_epoch} iters:")
            print(
                f"TRAIN --- | Loss: {np.mean(training_losses[-pseudo_epoch:]):.4} | OA: {np.mean(training_accs[-pseudo_epoch:]):.4} | F1M: {np.mean(training_f1ms[-pseudo_epoch:]):.4f}")
            print(
                f"VAL ----- | Loss: {np.mean(validation_losses[-pseudo_epoch:]):.4} | OA: {np.mean(validation_accs[-pseudo_epoch:]):.4} | F1M: {np.mean(validation_f1ms[-pseudo_epoch:]):.4f}")

        # If performance on validation set best so far, save model
        if np.mean(validation_f1ms[-pseudo_epoch:]) > best_val_f1:
            best_val_f1 = np.mean(validation_f1ms[-pseudo_epoch:])
            best_val_iter = current_iters
            if not is_local:
                torch.save(model.state_dict(), rf'./{model_use}_{reduce_method}_weights_{seed}.pt')

        # Step the scheduler based on the validation set performance
        current_iters += 1
        if current_iters > max_iters:
            stop_training = True
            break
        if current_iters > pseudo_epoch:
            scheduler.step(np.mean(validation_f1ms[-pseudo_epoch:]))
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != lr:
                print(f"Val f1 plateaued, lr {lr} -> {new_lr}")
                lr = new_lr
                lr_decreases.append(current_iters)
                if len(lr_decreases) >= 2:
                    stop_training = True
                    print("Val f1 decreased twice, ending training early")
                    break
    if stop_training: break

training_losses = np.array(training_losses);
validation_losses = np.array(validation_losses)
training_accs = np.array(training_accs);
validation_accs = np.array(validation_accs)
training_f1ms = np.array(training_f1ms);
validation_f1ms = np.array(validation_f1ms)
training_f1s = np.stack(training_f1s, axis=0);
validation_f1s = np.stack(validation_f1s, axis=0)
print(
    f"Training complete after {current_iters} iterations\n\ttotal samples       :    {current_iters * batch_size}\n\t -=-=-=-=-=-=-=-=-=-=-=-=-=-")
for cls_idx, samples_loaded in enumerate(dataset_train.total_sampled.numpy()):
    print(
        f"\t{annotation_class_names[cls_idx]}{(20 - len(annotation_class_names[cls_idx])) * ' '}:    {int(samples_loaded)}")
print(f"Metrics for final {pseudo_epoch} iterations:")
print(
    f"TRAIN --- | Loss: {training_losses[-pseudo_epoch:].mean():.4f} | OA: {training_accs[-pseudo_epoch:].mean():.4f} | f1: {training_f1ms[-pseudo_epoch:].mean():.4f}")
print(
    f"VAL ----- | Loss: {validation_losses[-pseudo_epoch:].mean():.4f} | OA: {validation_accs[-pseudo_epoch:].mean():.4f} | f1: {validation_f1ms[-pseudo_epoch:].mean():.4f}")

# test
running_loss_test = 0
test_preds, test_targets = [], []
if not is_local:
    model.load_state_dict(torch.load(rf'./{model_use}_{reduce_method}_weights_{seed}.pt', weights_only=True))

model.eval()
with torch.no_grad():
    for batch_idx, (data, label) in enumerate(test_loader):
        data = data.to(device)
        label = label.to(device)

        # Push through model
        out = model(data)
        loss = loss_fn(out, label)

        # Calculate metrics
        running_loss_test += loss.cpu().item()
        pred = out.argmax(dim=1).detach().cpu().numpy()
        actual = label.cpu().numpy()
        test_preds.extend(pred)
        test_targets.extend(actual)

test_targets = np.array(test_targets); test_preds = np.array(test_preds)
test_loss = running_loss_test / batch_idx
test_acc = accuracy_score(test_targets, test_preds)
test_f1m = f1_score(test_targets, test_preds, average='macro')
test_f1 = f1_score(test_targets, test_preds, average=None)

print("Metrics on entire testing set:")
print(f"TEST ---- | Loss: {test_loss:.4f} | OA: {test_acc:.4f} | f1: {test_f1m:.4f}")
for cls_idx, f1 in enumerate(test_f1):
    print(f"{annotation_class_names[cls_idx]}{(20 - len(annotation_class_names[cls_idx])) * ' '} : {f1:.4f}")
print("Total samples loaded for each class during TESTING")
for cls_idx, samples_loaded in enumerate(dataset_test.total_sampled.numpy()):
    print(f"{annotation_class_names[cls_idx]}{(20-len(annotation_class_names[cls_idx])) * ' '}:    {int(samples_loaded)}")

## Evaluation

# Plot overall loss curves as a moving average of batch metrics
def moving_average(a, n=3): # https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
    a = np.pad(a, ((n-1)//2,(n-1)//2 + ((n-1) % 2)), mode='edge')
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
fig,ax = plt.subplots(1,3,figsize=(16,5))
ax[0].plot(np.arange(0,len(moving_average(training_losses,n=1))),moving_average(training_losses,n=1),alpha=0.3,color='cornflowerblue')
ax[0].plot(np.arange(0,len(moving_average(training_losses,n=50))),moving_average(training_losses,n=50),alpha=1,color='cornflowerblue',label="train")
ax[0].plot(np.arange(0,len(moving_average(validation_losses,n=1))),moving_average(validation_losses,n=1),alpha=0.3,color='orange')
ax[0].plot(np.arange(0,len(moving_average(validation_losses,n=50))),moving_average(validation_losses,n=50),alpha=1,color='orange',label="validation")
ax[0].scatter(current_iters,test_loss,color='green',label="test",marker="x")
ax[0].set_title("Loss"); ax[0].legend()

ax[1].plot(np.arange(0,len(moving_average(training_accs,n=1))),moving_average(training_accs,n=1),alpha=0.3,color='cornflowerblue')
ax[1].plot(np.arange(0,len(moving_average(training_accs,n=50))),moving_average(training_accs,n=50),alpha=1,color='cornflowerblue',label="train")
ax[1].plot(np.arange(0,len(moving_average(validation_accs,n=1))),moving_average(validation_accs,n=1),alpha=0.3,color='orange')
ax[1].plot(np.arange(0,len(moving_average(validation_accs,n=50))),moving_average(validation_accs,n=50),alpha=1,color='orange',label="validation")
ax[1].scatter(current_iters,test_acc,color='green',label="test",marker="x")
ax[1].set_title("Accuracy"); ax[1].legend()

ax[2].plot(np.arange(0,len(moving_average(training_f1ms,n=1))),moving_average(training_f1ms,n=1),alpha=0.3,color='cornflowerblue')
ax[2].plot(np.arange(0,len(moving_average(training_f1ms,n=50))),moving_average(training_f1ms,n=50),alpha=1,color='cornflowerblue',label="train")
ax[2].plot(np.arange(0,len(moving_average(validation_f1ms,n=1))),moving_average(validation_f1ms,n=1),alpha=0.3,color='orange')
ax[2].plot(np.arange(0,len(moving_average(validation_f1ms,n=50))),moving_average(validation_f1ms,n=50),alpha=1,color='orange',label="validation")
ax[2].scatter(current_iters,test_f1m,color='green',label="test",marker="x")
ax[2].set_title("Macro F1 Score"); ax[2].legend()

ax[0].axvline(x=best_val_iter, ymin=0, ymax=1, color='red',alpha=0.3)
ax[1].axvline(x=best_val_iter, ymin=0, ymax=1, color='red',alpha=0.3)
ax[2].axvline(x=best_val_iter, ymin=0, ymax=1, color='red',alpha=0.3)

for lrd in lr_decreases:
    ax[0].axvline(x=lrd, ymin=0, ymax=1, color='grey',alpha=0.3)
    ax[1].axvline(x=lrd, ymin=0, ymax=1, color='grey',alpha=0.3)
    ax[2].axvline(x=lrd, ymin=0, ymax=1, color='grey',alpha=0.3)

plt.tight_layout()
if not is_local:
    plt.savefig(f'./loss_curve_{model_use}_{reduce_method}_{seed}.png'); plt.close(fig)

# Plot per-class f1 scores as a moving average per-batch
training_f1s = np.stack(training_f1s,axis=0)
validation_f1s = np.stack(validation_f1s,axis=0)
fig,ax = plt.subplots(2,3,figsize=(15,5)); ax = ax.flatten()
for cls in range(n_classes):
    ax[cls].plot(np.arange(0,len(moving_average(training_f1s[:,cls],n=1))),moving_average(training_f1s[:,cls],n=1),alpha=0.3,color='k')
    ax[cls].plot(np.arange(0,len(moving_average(training_f1s[:,cls],n=50))),moving_average(training_f1s[:,cls],n=50),alpha=1,color='k',label="train")
    ax[cls].plot(np.arange(0,len(moving_average(validation_f1s[:,cls],n=1))),moving_average(validation_f1s[:,cls],n=1),alpha=0.3,color=annotation_class_colors[cls]/255)
    ax[cls].plot(np.arange(0,len(moving_average(validation_f1s[:,cls],n=50))),moving_average(validation_f1s[:,cls],n=50),alpha=1,color=annotation_class_colors[cls]/255, label="validation")
    ax[cls].scatter(current_iters,test_f1[cls],color=annotation_class_colors[cls]/255,label="test",marker="x")
    ax[cls].set_ylim(ymin=0,ymax=1)
    for lrd in lr_decreases:
        ax[cls].axvline(x=lrd, ymin=0, ymax=1, color='grey',alpha=0.5)
    ax[cls].axvline(x=best_val_iter, ymin=0, ymax=1, color='red',alpha=0.3)
fig.suptitle("Class-specific F1 scores")
plt.tight_layout()
if not is_local:
    plt.savefig(f'./loss_curve_byclass_{model_use}_{reduce_method}_{seed}.png'); plt.close(fig)


## Save model and results

# save model
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

# close hdf5 files
dataset_train.close()
dataset_val.close()
dataset_test.close()
