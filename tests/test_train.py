from cellpose import io, models, train, metrics, plot
from pathlib import Path
from subprocess import check_output, STDOUT
import os, shutil
from glob import glob

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def test_class_train(data_dir):
    """
    Trains a Cellpose model on 2D imaging data and saves the trained model.

        This method processes imaging data located in the specified directory,
        trains a Cellpose model using the training images and their corresponding labels,
        and saves the trained model to the specified directory. It also manages
        temporary directories for model storage and ensures that any previous
        model files are removed before training.

        Args:
            data_dir: The directory path where the 2D imaging data is located,
                      containing a 'train' subdirectory with training data and
                      corresponding labels.

        Returns:
            None: The method does not return a value. It prints the path to the
                  saved model after training.
    """
    train_dir = str(data_dir.joinpath("2D").joinpath("train"))
    model_dir = str(data_dir.joinpath("2D").joinpath("train").joinpath("models"))
    shutil.rmtree(model_dir, ignore_errors=True)
    output = io.load_train_test_data(train_dir, mask_filter="_cyto_masks")
    images, labels, image_names, test_images, test_labels, image_names_test = output
    model = models.CellposeModel(pretrained_model=None, diam_mean=30)
    cpmodel_path = train.train_seg(
        model.net,
        images,
        labels,
        train_files=image_names,
        test_data=test_images,
        test_labels=test_labels,
        test_files=image_names_test,
        channels=[2, 1],
        save_path=train_dir,
        n_epochs=3,
    )[0]
    io.add_model(cpmodel_path)
    io.remove_model(cpmodel_path, delete=True)
    print(">>>> model trained and saved to %s" % cpmodel_path)


def test_cli_train(data_dir):
    """
    Trains a Cellpose model using the specified data directory.

        This method sets up and executes training commands for the Cellpose
        segmentation algorithm. It first prepares the training directory,
        removes any existing model directory, and runs the training command.
        After training, it identifies and loads the pretrained model for
        further processing.

        Args:
            data_dir: The directory containing the training data. It is expected
                that this directory has a '2D/train' subdirectory that includes
                the training files and masks.

        Raises:
            ValueError: If there is an error during the execution of the training
                command or while loading the pretrained model.

        Returns:
            None
    """
    # import sys
    # path_root = Path(__file__).parents[1]
    # sys.path.append(str(path_root))
    # print(Path(__file__).parents[0],Path(__file__).parents[1],Path(__file__).parents[2])
    train_dir = str(data_dir.joinpath("2D").joinpath("train"))
    model_dir = str(data_dir.joinpath("2D").joinpath("train").joinpath("models"))
    shutil.rmtree(model_dir, ignore_errors=True)
    cmd = (
        "python -m cellpose --train --train_size --n_epochs 3 --dir %s --mask_filter _cyto_masks --pretrained_model None --chan 2 --chan2 1 --diam_mean 40"
        % train_dir
    )
    try:
        cmd_stdout = check_output(cmd, stderr=STDOUT, shell=True).decode()
    except Exception as e:
        print(e)
        raise ValueError(e)

    model_dir = data_dir.joinpath("2D").joinpath("train").joinpath("models")
    print(model_dir)
    pretrained_models = model_dir.glob("*")
    pretrained_models = [os.fspath(pmodel.absolute()) for pmodel in pretrained_models]
    print(pretrained_models)
    pretrained_model = [
        pmodel for pmodel in pretrained_models if pmodel[-9:] != "_size.npy"
    ][0]
    print(pretrained_model)
    cmd = (
        "python -m cellpose --dir %s --pretrained_model %s --chan 2 --chan2 1 --diam_mean 40"
        % (train_dir, pretrained_model)
    )
    try:
        cmd_stdout = check_output(cmd, stderr=STDOUT, shell=True).decode()
    except Exception as e:
        print(e)
        raise ValueError(e)


def test_cli_train_pretrained(data_dir):
    """
    Trains a pretrained model using the Cellpose CLI.

        This method constructs and executes a command to train a pretrained Cellpose model on a specified dataset directory.
        It ensures any existing model directory is removed before initiating the training process.

        Args:
            data_dir: The root directory containing subdirectories for training data.

        Returns:
            None: This method does not return a value. It raises a ValueError if an exception occurs during the command execution.
    """
    train_dir = str(data_dir.joinpath("2D").joinpath("train"))
    model_dir = str(data_dir.joinpath("2D").joinpath("train").joinpath("models"))
    shutil.rmtree(model_dir, ignore_errors=True)
    cmd = (
        "python -m cellpose --train --train_size --n_epochs 3 --dir %s --mask_filter _cyto_masks --pretrained_model cyto --chan 2 --chan2 1"
        % train_dir
    )
    try:
        cmd_stdout = check_output(cmd, stderr=STDOUT, shell=True).decode()
    except Exception as e:
        print(e)
        raise ValueError(e)
