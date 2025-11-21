import numpy as np
import torch
from torch import nn
import h5py
import cv2
from torchvision.transforms import v2

# Torch dataset to sample random patches from full set
# Makes list of every annotated pixel, then samples from this list to index into dataset
# This has high initial cost, but subsequent loads are cheap in a chunked hdf5 dataset;
# Entire cores do not have to be loaded to sample a single small patch, for example
class ftir_patching_dataset(torch.utils.data.Dataset):
    def __init__(self,
            hdf5_filepaths,
            mask_filepaths,
            annotation_filepaths,
            annotation_class_names,
            annotation_class_colors,
            patch_dim=25, image_size=256, augment=True,
    ):

        # Define data paths
        self.hdf5_filepaths = hdf5_filepaths
        self.mask_filepaths = mask_filepaths
        self.annotation_filepaths = annotation_filepaths
        self.augment = augment

        # patch dimensions
        self.image_size = image_size
        self.patch_dim = patch_dim
        self.patch_minus = patch_dim // 2;
        self.patch_plus = 1 + (patch_dim // 2)

        # class data
        self.annotation_class_colors = annotation_class_colors
        self.annotation_class_names = annotation_class_names
        self.total_sampled = torch.zeros(len(self.annotation_class_colors))

        # define data augmentation pipeline
        self.transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
        ])

        # Open every core hdf5 file
        self.open()

    def __len__(self):
        return self.total_pixels

    def __getitem__(self, idx):
        # get patch data
        row = self.rows[idx]
        col = self.cols[idx]
        cidx = self.cidxs[idx]
        label = self.tissue_classes[idx]
        self.total_sampled[label] += 1

        # Are dimensions of patch okay
        idx_u = row - self.patch_minus
        idx_d = row + self.patch_plus
        idx_l = col - self.patch_minus
        idx_r = col + self.patch_plus
        pad_u = max(-idx_u, 0);
        idx_u = max(idx_u, 0)
        pad_d = max(idx_d - self.image_size, 0);
        idx_d = min(idx_d, self.image_size)
        pad_l = max(-idx_l, 0);
        idx_l = max(idx_l, 0)
        pad_r = max(idx_r - self.image_size, 0);
        idx_r = min(idx_r, self.image_size)

        # get patch
        patch = torch.from_numpy(
            self.hdf5_files[cidx]['spectra'][idx_u:idx_d, idx_l:idx_r, :],
        ).permute(2, 0, 1)
        patch *= torch.from_numpy(
            self.hdf5_files[cidx]['mask'][idx_u:idx_d, idx_l:idx_r,],
        ).unsqueeze(0)

        # pad patch
        patch = torch.nn.functional.pad(patch, (pad_l, pad_r, pad_u, pad_d, 0, 0))

        if self.augment:
            patch = self.transforms(patch)
        return patch, label

    # split annotations from H x W x 3 to C x H x W, one/zerohot along C dimension
    def split_annotations(self, annotations_img):
        split = torch.zeros((len(self.annotation_class_colors), *annotations_img.shape[:-1]))
        for c, col in enumerate(self.annotation_class_colors):
            split[c, :, :] = torch.from_numpy(np.all(annotations_img == self.annotation_class_colors[c], axis=-1))
        return split

    # open every file
    def open(self):
        self.hdf5_files = []
        self.tissue_classes = []
        self.rows = []
        self.cols = []
        self.cidxs = []

        # for every core in dataset,
        for cidx in range(0, len(self.hdf5_filepaths)):
            # open annotations and remove edges and non-tissue px
            annotation = self.split_annotations(cv2.imread(self.annotation_filepaths[cidx])[:, :, ::-1])
            mask = torch.from_numpy(cv2.imread(self.mask_filepaths[cidx])[:, :, 1]) / 255
            annotation *= mask
            # for every class,
            for cls in range(len(self.annotation_class_names)):
                # get location of annotations, append to lists
                r, c = torch.where(annotation[cls])
                num_cls = annotation[cls].sum().int().item()
                self.tissue_classes.extend([cls, ] * num_cls)
                self.cidxs.extend([cidx, ] * num_cls)
                self.rows.extend(r)
                self.cols.extend(c)
            # add open hdf5 file to list
            self.hdf5_files.append(h5py.File(self.hdf5_filepaths[cidx], 'r'))

        # construct data tensors
        self.rows = torch.Tensor(self.rows).int()
        self.cols = torch.Tensor(self.cols).int()
        self.tissue_classes = torch.Tensor(self.tissue_classes).long()
        self.cidxs = torch.Tensor(self.cidxs).int()
        self.total_pixels = len(self.cidxs)

    # close every open hdf5 file
    def close(self):
        for cidx in range(len(self.hdf5_files)):
            self.hdf5_files[cidx].close()
        self.hdf5_files = []
        self.tissue_classes = []
        self.xs = []
        self.ys = []

# Torch dataset that loads entire cores at once
# This is a much simpler dataset class, more akin to a traditional image dataset
class ftir_core_dataset(torch.utils.data.Dataset):
    def __init__(self,
            hdf5_filepaths,
            mask_filepaths,
            annotation_filepaths,
            annotation_class_names,
            annotation_class_colors,
            augment=False):
        self.hdf5_filepaths = hdf5_filepaths
        self.mask_filepaths = mask_filepaths
        self.annotation_filepaths = annotation_filepaths
        self.augment = augment

        # class data
        self.annotation_class_colors = annotation_class_colors
        self.annotation_class_names = annotation_class_names

    def __len__(self):
        return len(self.hdf5_filepaths)

    # split annotations from H x W x 3 to C x H x W, one/zerohot along C dimension
    def split_annotations(self, annotations_img):
        split = torch.zeros((len(self.annotation_class_colors), *annotations_img.shape[:-1]))
        for c, col in enumerate(self.annotation_class_colors):
            split[c, :, :] = torch.from_numpy(np.all(annotations_img == self.annotation_class_colors[c], axis=-1))
        return split

    def __getitem__(self, idx):

        # open hdf5 file
        hdf5_file = h5py.File(self.hdf5_filepaths[idx], 'r')

        # get mask
        mask = torch.from_numpy(
            hdf5_file['mask'][:],
        ).unsqueeze(0)

        # get ftir
        ftir = torch.from_numpy(
            hdf5_file['spectra'][:],
        ).permute(2, 0, 1)
        hdf5_file.close()
        ftir *= mask

        # get annotations
        annotations = self.split_annotations(cv2.imread(self.annotation_filepaths[idx])[:, :, ::-1])
        annotations *= mask
        has_annotations = annotations.sum(dim=0) != 0

        if self.augment:
            to_aug = torch.rand((2,))
            if to_aug[0] > 0.5:  # hflip
                ftir = torch.flip(ftir, (-1,))
                annotations = torch.flip(annotations, (-1,))
                has_annotations = torch.flip(has_annotations, (-1,))
                mask = torch.flip(mask, (-1,))
            if to_aug[1] > 0.5:  # vflip
                ftir = torch.flip(ftir, (-2,))
                annotations = torch.flip(annotations, (-2,))
                has_annotations = torch.flip(has_annotations, (-2,))
                mask = torch.flip(mask, (-2,))

        return ftir, annotations, mask, has_annotations

class LinearReduction(nn.Module):
    def __init__(self, input_dim, reduce_dim):
        super().__init__()
        self.reduce_dim = reduce_dim
        self.input_norm = nn.BatchNorm2d(input_dim)
        self.projection = nn.Conv2d(input_dim, reduce_dim, kernel_size=1, stride=1)
        self.projection_norm = nn.BatchNorm2d(reduce_dim)

    def forward(self, x):
        return self.projection_norm(self.projection(self.input_norm(x)))

class FixedReduction(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_norm = nn.BatchNorm2d(input_dim)

    def forward(self, x):
        return self.input_norm(x)

