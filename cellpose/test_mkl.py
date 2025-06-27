"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import os, sys

os.environ["MKLDNN_VERBOSE"] = "1"
import numpy as np
import time

try:
    import mxnet as mx

    x = mx.sym.Variable("x")
    MXNET_ENABLED = True
except:
    MXNET_ENABLED = False


def test_mkl():
    """
    Test the MKL (Math Kernel Library) integration with MXNet.

        This method performs a convolution operation using MXNet's symbolic API
        with randomly generated input and weights. It checks if MXNet is enabled
        and then sets up the necessary parameters for a convolution operation,
        binds the parameters to the execution context, and executes the forward
        pass to obtain the output.

        Parameters:
            None

        Returns:
            numpy.ndarray: The output of the convolution operation as a NumPy array.
    """
    if MXNET_ENABLED:
        num_filter = 32
        kernel = (3, 3)
        pad = (1, 1)
        shape = (32, 32, 256, 256)

        x = mx.sym.Variable("x")
        w = mx.sym.Variable("w")
        y = mx.sym.Convolution(
            data=x,
            weight=w,
            num_filter=num_filter,
            kernel=kernel,
            no_bias=True,
            pad=pad,
        )
        exe = y.simple_bind(mx.cpu(), x=shape)

        exe.arg_arrays[0][:] = np.random.normal(size=exe.arg_arrays[0].shape)
        exe.arg_arrays[1][:] = np.random.normal(size=exe.arg_arrays[1].shape)

        exe.forward(is_train=False)
        o = exe.outputs[0]
        t = o.asnumpy()


if __name__ == "__main__":
    if MXNET_ENABLED:
        test_mkl()
