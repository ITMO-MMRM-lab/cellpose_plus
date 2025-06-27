import numpy as np
import pytest

from cellpose.io import imread
from cellpose.transforms import normalize_img, random_rotate_and_resize, resize_image


@pytest.fixture
def img_3d(data_dir):
    """Fixture to load 3D image data for tests."""
    img = imread(str(data_dir.joinpath("3D").joinpath("rgb_3D.tif")))
    return img.transpose(0, 2, 3, 1).astype("float32")


@pytest.fixture
def img_2d(data_dir):
    """Fixture to load 2D image data for tests."""
    return imread(str(data_dir.joinpath("2D").joinpath("rgb_2D_tif.tif")))


def test_random_rotate_and_resize__default():
    """
    Tests the random rotation and resizing of a batch of images.

        This method generates a batch of random images, each of size 64x64,
        and applies the `random_rotate_and_resize` function to this batch.

        Parameters:
            None

        Returns:
            None
    """
    nimg = 2
    X = [np.random.rand(64, 64) for i in range(nimg)]
    random_rotate_and_resize(X)


def test_normalize_img(img_3d):
    """
    Tests the normalization of a 3D image by applying various normalization methods.

    This method runs several assertions to ensure that the normalized image maintains the
    same shape as the original input image when different normalization parameters are applied.

    Args:
        img_3d: The input 3D image to be normalized.

    Returns:
        None
    """
    img_norm = normalize_img(img_3d, norm3D=True)
    assert img_norm.shape == img_3d.shape

    img_norm = normalize_img(img_3d, norm3D=True, tile_norm_blocksize=25)
    assert img_norm.shape == img_3d.shape

    img_norm = normalize_img(img_3d, norm3D=False, sharpen_radius=8)
    assert img_norm.shape == img_3d.shape


def test_normalize_img_with_lowhigh_and_invert(img_3d):
    """
    Tests the normalization of a 3D image array using specified low and high values.

        This method evaluates the normalization of a 3D image by asserting
        the conditions on the normalized values using various ranges for low
        and high. It checks normalization both for the entire image and
        for individual channels.

        Args:
            img_3d: A 3D image array that is to be normalized.

        Returns:
            None: This method does not return a value; it raises an assertion error if the conditions are not met.
    """
    img_norm = normalize_img(img_3d, lowhigh=(img_3d.min() + 1, img_3d.max() - 1))
    assert img_norm.min() < 0 and img_norm.max() > 1

    img_norm = normalize_img(img_3d, lowhigh=(img_3d.min(), img_3d.max()))
    assert 0 <= img_norm.min() < img_norm.max() <= 1

    img_norm_channelwise = normalize_img(
        img_3d,
        lowhigh=(
            (img_3d[..., 0].min(), img_3d[..., 0].max()),
            (img_3d[..., 1].min(), img_3d[..., 1].max()),
        ),
    )
    assert img_norm_channelwise.min() >= 0 and img_norm_channelwise.max() <= 1


def test_normalize_img_exceptions(img_3d):
    """
    Test exceptions raised by the normalize_img function.

        This method tests the normalize_img function by passing
        various invalid inputs to ensure it raises the appropriate
        ValueError exceptions. It specifically checks scenarios where
        the input image is not 2-dimensional, the lowhigh parameters
        have inconsistent shapes, and other invalid parameter configurations.

        Args:
            img_3d: A 3-dimensional image array that is tested for normalization exceptions.

        Raises:
            ValueError: If the normalization conditions are invalid, as expected during the tests.

        Returns:
            None
    """
    img_2D = img_3d[0, ..., 0]
    with pytest.raises(ValueError):
        normalize_img(img_2D)

    with pytest.raises(ValueError):
        normalize_img(img_3d, lowhigh=(0, 1, 2))

    with pytest.raises(ValueError):
        normalize_img(img_3d, lowhigh=((0, 1), (0, 1, 2)))

    with pytest.raises(ValueError):
        normalize_img(img_3d, lowhigh=((0, 1),) * 4)

    with pytest.raises(ValueError):
        normalize_img(img_3d, percentile=(1, 101))

    with pytest.raises(ValueError):
        normalize_img(
            img_3d, lowhigh=None, tile_norm_blocksize=0, normalize=False, invert=True
        )


def test_resize(img_2d):
    """
    Tests the resizing of a given 2D image to specified dimensions.

        This method tests the functionality of the image resizing utility by resizing
        the input image (img_2d) to both 100x200 pixels at different bit depths (uint8,
        uint16, and uint32). It checks if the resized images have the correct shape
        and data type.

        Args:
            img_2d: The input 2D image to be resized, which should be in a format
                     that can be converted to uint8, uint16, and uint32.

        Returns:
            None: This method does not return a value; it asserts conditions to validate
            the properties of the resized images.
    """
    Lx = 100
    Ly = 200

    img8 = resize_image(img_2d.astype("uint8"), Lx=Lx, Ly=Ly)
    assert img8.shape == (Ly, Lx, 3)
    assert img8.dtype == np.uint8

    img16 = resize_image(img_2d.astype("uint16"), Lx=Lx, Ly=Ly)
    assert img16.shape == (Ly, Lx, 3)
    assert img16.dtype == np.uint16

    img32 = resize_image(img_2d.astype("uint32"), Lx=Lx, Ly=Ly)
    assert img32.shape == (Ly, Lx, 3)
    assert img32.dtype == np.uint32
