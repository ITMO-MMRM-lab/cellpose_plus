from cellpose import io, denoise
from pathlib import Path
from subprocess import check_output, STDOUT
import os, shutil
import numpy as np


def clear_output(data_dir, image_names):
    """
    Removes specific output files associated with given image names.

        This method constructs file paths for output files based on the provided directory and image names,
        and deletes the output files if they exist. The method handles both 2D and 3D images differently
        based on their naming convention.

        Args:
            data_dir: The directory containing the images, which will be used to construct paths for output files.
            image_names: A list of image names which are used to determine the corresponding output files to remove.

        Returns:
            None
    """
    data_dir_2D = data_dir.joinpath("2D")
    data_dir_3D = data_dir.joinpath("2D")
    for image_name in image_names:
        if "2D" in image_name:
            cached_file = str(data_dir_2D.joinpath(image_name))
            ext = ".png"
        else:
            cached_file = str(data_dir_3D.joinpath(image_name))
            ext = ".tif"
        name, ext = os.path.splitext(cached_file)
        for rtype in ["denoise_cyto3", "deblur_cyto3", "upsample_cyto3"]:
            output = name + f"_{rtype}.tif"
            if os.path.exists(output):
                os.remove(output)


def test_class_2D(data_dir, image_names):
    """
    Tests various denoising, deblurring, and upsampling models on a 2D image.

        This method evaluates multiple image processing models on a specified 2D image,
        checking the output image's shape against expected dimensions. The results are saved
        to disk.

        Args:
            data_dir: The directory where the image is located and where results will be saved.
            image_names: A list of image names to be processed.

        Returns:
            None
    """
    clear_output(data_dir, image_names)
    image_name = "gray_2D.png"
    img = io.imread(str(data_dir.joinpath("2D").joinpath(image_name)))
    model_types = ["denoise_cyto3", "deblur_cyto3", "upsample_cyto3"]
    chan = [2, 1, 0]
    chan2 = [1, 0, 0]
    diams = [30.0, 30.0, 15.0]
    shapes = [
        (*img.shape[:2], 1),
        (*img.shape[:2], 1),
        (img.shape[0] * 2, img.shape[1] * 2, 1),
    ]
    for m, model_type in enumerate(model_types):
        model = denoise.DenoiseModel(model_type=model_type, chan2=True)
        img_restore = model.eval(img, diameter=diams[m], channels=[chan[m], chan2[m]])
        assert img_restore.shape == shapes[m]
        io.imsave(
            str(data_dir.joinpath("2D").joinpath(f"gray_2D_{model_type}.tif")),
            img_restore,
        )
    clear_output(data_dir, image_names)


def test_dn_cp_class_2D(data_dir, image_names):
    """
    Tests the Cellpose DenoiseModel for 2D image processing.

        This method processes a 2D RGB image by applying various models of the Cellpose DenoiseModel, and saves the processed images.

        Args:
            data_dir: The directory containing the input images and where the output images will be saved.
            image_names: A list of image names to be processed.

        Returns:
            None: This method does not return a value. It saves the processed images to the specified directory.
    """
    clear_output(data_dir, image_names)
    image_name = "rgb_2D.png"
    img = io.imread(str(data_dir.joinpath("2D").joinpath(image_name)))
    model_types = ["denoise_cyto3", "deblur_cyto3", "upsample_cyto3"]
    chan = [2, 1, 0]
    chan2 = [1, 0, 0]
    diams = [30.0, 30.0, 15.0]
    shapes = [
        (*img.shape[:2], 2),
        (*img.shape[:2], 1),
        (img.shape[0] * 2, img.shape[1] * 2, 1),
    ]
    for m, model_type in enumerate(model_types):
        model = denoise.CellposeDenoiseModel(
            model_type="cyto3", restore_type=model_type, chan2_restore=True
        )
        masks, flows, styles, img_restore = model.eval(
            img, diameter=diams[m], channels=[chan[m], chan2[m]]
        )
        assert img_restore.shape == shapes[m]
        assert masks.shape == shapes[m][:2]
        io.imsave(
            str(data_dir.joinpath("2D").joinpath(f"rgb_2D_{model_type}.tif")),
            img_restore,
        )
    clear_output(data_dir, image_names)


def test_cli_2D(data_dir, image_names):
    """
    Run the Cellpose model in 2D using specified parameters.

    This method executes a command-line interface (CLI) command to run the Cellpose
    image segmentation model on 2D images. It clears previous outputs, constructs a
    command with the necessary parameters, and executes it while handling any errors
    that may occur during the execution.

    Args:
        data_dir: The directory where the image data is located.
        image_names: The names of the images to be processed.

    Returns:
        None: This method does not return a value, but it prints the command output
        or raises a ValueError if an error occurs during execution.
    """
    clear_output(data_dir, image_names)
    model_types = ["denoise_cyto3"]
    chan = [2]
    chan2 = [1]
    for m, model_type in enumerate(model_types):
        cmd = (
            "python -m cellpose --dir %s --pretrained_model %s --restore_type %s --chan %d --chan2 %d --chan2_restore --diameter 30"
            % (str(data_dir.joinpath("2D")), "cyto3", model_type, chan[m], chan2[m])
        )
        try:
            cmd_stdout = check_output(cmd, stderr=STDOUT, shell=True).decode()
            print(cmd_stdout)
        except Exception as e:
            print(e)
            raise ValueError(e)
        clear_output(data_dir, image_names)
