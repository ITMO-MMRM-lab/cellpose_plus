def test_cellpose_imports_without_error():
    """
    Tests the importing of the Cellpose library without errors.

        This method attempts to import the Cellpose library and instantiates
        a CellposeModel to verify that the necessary components can be
        successfully loaded without any import errors.

        Returns:
            None: This method does not return any value. It raises an
            exception if the imports fail.
    """
    import cellpose
    from cellpose import models, core

    model = models.CellposeModel()


def test_model_zoo_imports_without_error():
    """
    Test the importing of models from the Cellpose library without errors.

        This method attempts to import various model types from the Cellpose library
        and ensures that no errors occur during the import process. Specifically, it
        instantiates models that do not contain the terms 'neurips' or 'transformer'
        in their names.

        Returns:
            None: This method does not return any value, but raises an exception if
            an import error occurs.
    """
    from cellpose import models, denoise

    for model_name in models.MODEL_NAMES:
        if "neurips" not in model_name and "transformer" not in model_name:
            model = models.CellposeModel(model_type=model_name)


def test_gui_imports_without_error():
    """
    Test the import of the GUI without raising any errors.

        This method attempts to import the GUI module from the cellpose package
        to ensure that the import works correctly without throwing any exceptions.

        Returns:
            None: This method does not return a value. It is meant to verify that
            the import operation is successful.
    """
    from cellpose import gui


def test_gpu_check():
    """
    Check and enable GPU usage for the Cellpose model.

        This method verifies if the GPU can be utilized by the Cellpose library
        and configures the library to use it accordingly.

        Returns:
            None: The function does not return any value. It only modifies the
            GPU setting for the Cellpose library if possible.
    """
    #     from cellpose import models
    #     models.use_gpu()
    from cellpose import core

    core.use_gpu()


def test_model_dir():
    """
    Tests the model directory and validates the shape of the output masks.

        This method sets up the environment for the Cellpose library by defining
        the local models path. It then initializes a Cellpose model using a
        pretrained model and evaluates it with a randomly generated input.
        Finally, it asserts that the output mask has the expected shape.

        Parameters:
            None

        Returns:
            None: This method does not return a value but will raise an AssertionError
            if the output mask's shape does not match the expected dimensions of
            (224, 224).
    """
    import os, pathlib
    import numpy as np

    os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = os.fspath(
        pathlib.Path.home().joinpath(".cellpose")
    )

    from cellpose import models

    model = models.CellposeModel(pretrained_model="cyto3")
    masks = model.eval(np.random.randn(224, 224))[0]
    assert masks.shape == (224, 224)
