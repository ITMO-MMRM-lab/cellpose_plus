import numpy as np
import pytest
import torch

from cellpose.dynamics import masks_to_flows_gpu

CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="No CUDA device available")
def test__masks_to_flows_gpu__single_object():
    """
    Tests the masks_to_flows_gpu function with a single object using CUDA.

        This method creates a binary mask of a single object and applies the
        masks_to_flows_gpu function to generate optical flow values on a
        CUDA-enabled device. The test is skipped if no CUDA device is
        available.

        Parameters:
            None

        Returns:
            None
    """
    masks = np.zeros((32, 32), dtype=int)
    masks[16:18, 16:18] = 1
    masks_to_flows_gpu(masks, device=torch.device("cuda"))
