"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import time
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from pathlib import Path
from cellpose import transforms, io, metrics
from cellpose.models import CellposeModel

# uses torch
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from torch.nn import MSELoss
from torch.optim import Adam

# PATH TO REPO
import sys

sys.path.append("/github/noise2self/")
from mask import Masker
from models.unet import Unet

device = torch.device("cuda")


class Cells(Dataset):
    """
    A class to manage and transform cell image data for analysis.

    This class initializes with input data and applies various transformations
    to prepare the data for further processing. Transformations include random
    rotation, resized cropping, horizontal flipping, and type conversion.

    Methods:
        __init__(): Initializes an instance of the class with data and cell transformations.
        __len__(): Retrieves the number of elements in the data attribute.
        __getitem__(): Retrieve an item from the dataset at a specified index.

    Attributes:
        data: The input data to be processed by the instance.
        xy: A tuple representing the target size for the random cropped images.

    The `__init__` method sets up the initial data and the transformations to be applied.
    The `__len__` method returns the total number of elements in the data attribute.
    The `__getitem__` method applies the defined transformations to return the
    transformed image at the specified index.
    """

    def __init__(self, data, xy=(128, 128)):
        """
        Initializes an instance of the class with data and cell transformations.

            This method sets up the initial data and defines a series of transformations to be applied to
            the data, including random rotation, resized cropping, horizontal flipping, and type conversion.

            Args:
                data: The input data to be processed by the instance.
                xy: A tuple representing the target size for the random cropped images, with
                    default values set to (128, 128).

            Returns:
                None
        """
        self.data = data
        self.cell_transforms = v2.Compose(
            [
                v2.RandomRotation(degrees=180),
                v2.RandomResizedCrop(
                    size=xy, scale=(0.75, 1.25), ratio=(1.0, 1.0), antialias=True
                ),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

    def __len__(self):
        """
        Retrieves the number of elements in the data attribute.

            This method returns the length of the data attribute, which is expected
            to be a collection.

            Returns:
                The number of elements in the data attribute.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Retrieve an item from the dataset at a specified index.

            This method applies cell transformations to the data at the given index
            and returns the transformed image.

            Args:
                index: The index of the item to retrieve from the dataset.

            Returns:
                A transformed image retrieved from the dataset.
        """
        img = self.cell_transforms(torch.from_numpy(self.data[index]).to(device))
        return img


def train_per_image(img_noisy):
    """
    Trains a U-Net model on a given noisy image to denoise it.

        This method processes a single noisy image, training a U-Net model
        using mean squared error loss to minimize the difference between
        the denoised output and the original noisy image. The training
        is conducted over multiple epochs with batch processing of the
        image data.

        Args:
            img_noisy: The input noisy image to be denoised.

        Returns:
            The denoised version of the input image.
    """

    masker = Masker(width=4, mode="interpolate")
    torch.cuda.manual_seed(0)
    model = Unet().to(device)

    loss_function = MSELoss()
    optimizer = Adam(model.parameters(), lr=5e-4)

    cells_train = Cells(
        np.tile(img_noisy[np.newaxis, ...], (8, 1, 1, 1)), xy=(128, 128)
    )
    data_loader = DataLoader(cells_train, batch_size=8, shuffle=True)

    for ep in range(100):
        for k, batch in enumerate(data_loader):
            noisy_images = batch

            net_input, mask = masker.mask(noisy_images, k)
            net_output = model(net_input)

            loss = loss_function(net_output * mask, noisy_images * mask)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

    img = img_noisy.copy()
    img, ysub, xsub = transforms.pad_image_ND(img)
    img = torch.from_numpy(img).to(device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        simple_output = model(img)
    out = (
        simple_output.squeeze()[ysub[0] : ysub[-1] + 1, xsub[0] : xsub[-1] + 1]
        .cpu()
        .numpy()
    )

    return out


def train_per_image_synthetic(root, ctype="cyto2", plot=False, save=True):
    """
    Trains a model on synthetic noisy images and generates segmentation masks.

        This method processes synthetic noisy images to produce segmented outputs using a specified
        model type. It optionally plots the noisy images, processed images, and segmentation masks
        for visual inspection, and can save the results to a file.

        Args:
            root: The root directory where the noisy test images are stored.
            ctype: The type of model to use for segmentation. Default is "cyto2".
            plot: A boolean indicating whether to plot the images and masks during processing.
                  Default is False.
            save: A boolean indicating whether to save the output data to a file. Default is True.

        Returns:
            A tuple containing:
                - A list of processed images.
                - A list of corresponding segmentation masks.
    """
    noise_type = "poisson"

    dat = np.load(
        root / "noisy_test" / f"test_{noise_type}.npy", allow_pickle=True
    ).item()
    test_noisy = dat["test_noisy"]
    test_labels = dat["masks_true"]
    diam_test = (
        dat["diam_test"] if "diam_test" in dat else 30.0 * np.ones(len(test_noisy))
    )

    imgs_n2s, masks_n2s = [], []

    seg_model = CellposeModel(gpu=True, model_type=f"{ctype}")
    for i in trange(len(test_noisy)):
        out = train_per_image(test_noisy[i])

        masks = seg_model.eval(
            out, diameter=diam_test[i], channels=[1, 0], channel_axis=0, normalize=True
        )[0]

        masks_n2s.append(masks)
        imgs_n2s.append(out)

        if plot:
            print(f">>> IMAGE {i}, n_masks = {masks.max()}")
            plt.figure(figsize=(12, 3))
            plt.subplot(1, 4, 1)
            plt.imshow(test_noisy[i][0])
            plt.subplot(1, 4, 2)
            plt.imshow(out)
            plt.subplot(1, 4, 3)
            plt.imshow(masks)
            plt.subplot(1, 4, 4)
            plt.imshow(test_labels[i])
            plt.show()

    dat["masks_n2s"] = masks_n2s
    dat["test_n2s"] = imgs_n2s
    if save:
        np.save(root / "noisy_test" / f"test_{noise_type}_n2s.npy", dat)

    return imgs_n2s, masks_n2s


def train_test_specialist(root, n_epochs=50, lr=5e-4, test=True):
    """
    Trains a U-Net model on noisy cell images and optionally tests it
        on a validation set, returning predicted masks and average precision.

        This method prepares training and validation datasets by loading images
        and corresponding masks. It then trains a U-Net model for a specified
        number of epochs, monitoring the loss on both training and validation
        datasets. If requested, it performs testing on a set of noisy test images
        and generates segmentation masks, returning the masks and average precision
        metrics.

        Args:
            root: The root directory containing the dataset.
            n_epochs: The number of epochs to train the model. Defaults to 50.
            lr: The learning rate for the optimizer. Defaults to 5e-4.
            test: A boolean indicating whether to perform testing after training.
                  Defaults to True.

        Returns:
            If `test` is True, returns a tuple containing:
                - imgs: A list of images generated by the model.
                - masks_n2s: A list of predicted masks for the images.
                - ap: Average precision metrics for the predicted masks compared to the true masks.
            If `test` is False, returns the validation loss.
    """
    n_train = 3 * 89 * 20
    n_val = 89 * 20

    dat = np.load(root / "noisy_test" / "test_poisson.npy", allow_pickle=True).item()
    test_noisy = dat["test_noisy"][:11]
    masks_true = dat["masks_true"][:11]
    diam_test = dat["diam_test"]

    im_train = [
        io.imread(Path(root / "noisy_test" / "care" / "source" / f"{i:03d}.tif"))[
            np.newaxis, :, :
        ]
        for i in range(n_train)
    ]
    im_train.extend(test_noisy)
    im_val = [
        io.imread(Path(root / "noisy_test" / "care" / "source" / f"{i:03d}.tif"))
        for i in range(n_train, n_train + n_val)
    ]

    cells_train = Cells(im_train, xy=(128, 128))
    cells_val = Cells(np.array(im_val)[:, np.newaxis, :, :], xy=(128, 128))

    data_loader = DataLoader(cells_train, batch_size=64, shuffle=True)
    val_loader = DataLoader(cells_val, batch_size=64, shuffle=True)

    loss_function = MSELoss()
    masker = Masker(width=4, mode="interpolate")

    model = Unet().to(device)
    tic = time.time()
    optimizer = Adam(model.parameters(), lr=lr)
    for ep in range(n_epochs):
        model.train()
        train_loss = 0
        for i, batch in enumerate(data_loader):
            noisy_images = batch
            net_input, mask = masker.mask(noisy_images, i)
            net_output = model(net_input)
            loss = loss_function(net_output * mask, noisy_images * mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += (loss.item()) * len(batch)
        train_loss /= len(cells_train) / len(cells_val)
        if ep < 10 or ep % 5 == 0 or ep == n_epochs - 1:
            val_loss = 0
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    noisy_images = batch
                    net_input, mask = masker.mask(noisy_images, i)
                    net_output = model(net_input)
                    loss = loss_function(net_output * mask, noisy_images * mask)
                    val_loss += (loss.item()) * len(batch)
            # val_loss /= len(cells_val)
            print(
                f"Loss ( {ep} ): \t train: {train_loss:.3f}, val: {val_loss:.3f}, {time.time()-tic:.2f}s"
            )
    if not test:
        return val_loss
    else:

        imgs = []
        masks_n2s = []
        for i in range(len(test_noisy)):
            img = test_noisy[i].copy()
            img, ysub, xsub = transforms.pad_image_ND(img)
            img = torch.from_numpy(img).to(device).unsqueeze(0)
            model.eval()
            with torch.no_grad():
                simple_output = model(img)
            out = (
                simple_output.squeeze()[ysub[0] : ysub[-1] + 1, xsub[0] : xsub[-1] + 1]
                .cpu()
                .numpy()
            )

            imgs.append(out)
            seg_model = CellposeModel(gpu=True, model_type="cyto2")

            masks = seg_model.eval(
                out,
                diameter=diam_test[i],
                channels=[1, 0],
                channel_axis=0,
                normalize=True,
            )[0]

            masks_n2s.append(masks)

        for i in range(11):
            assert masks_true[i].shape == masks_n2s[i].shape

        ap, tp, fp, fn = metrics.average_precision(masks_true, masks_n2s)
        print(ap.mean(axis=0))

        dat[f"test_n2s"] = imgs
        dat[f"masks_n2s"] = masks_n2s

        np.save(root / "noisy_test" / f"test_poisson_n2s_specialist.npy", dat)

        return imgs, masks_n2s, ap
