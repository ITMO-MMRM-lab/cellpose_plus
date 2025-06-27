"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def batchconv(in_channels, out_channels, sz, conv_3D=False):
    """
    Creates a sequential model containing batch normalization, ReLU activation,
        and a convolutional layer.

        This method generates a neural network block that starts with batch normalization
        followed by a ReLU activation function and ends with either a 2D or 3D convolution
        layer based on the provided flag.

        Args:
            in_channels: The number of input channels to the convolutional layer.
            out_channels: The number of output channels from the convolutional layer.
            sz: The size of the convolutional kernel.
            conv_3D: A boolean flag indicating whether to use 3D convolution (True)
                      or 2D convolution (False).

        Returns:
            A sequential container comprising the batch normalization layer,
            ReLU activation, and the specified convolutional layer.
    """
    conv_layer = nn.Conv3d if conv_3D else nn.Conv2d
    batch_norm = nn.BatchNorm3d if conv_3D else nn.BatchNorm2d
    return nn.Sequential(
        batch_norm(in_channels, eps=1e-5, momentum=0.05),
        nn.ReLU(inplace=True),
        conv_layer(in_channels, out_channels, sz, padding=sz // 2),
    )


def batchconv0(in_channels, out_channels, sz, conv_3D=False):
    """
    Creates a sequential batch normalization and convolutional layer.

        This method constructs a neural network module consisting of
        a batch normalization layer followed by a convolutional layer.
        The type of convolutional layer (2D or 3D) is determined by the
        `conv_3D` parameter.

        Args:
            in_channels: The number of input channels for the network.
            out_channels: The number of output channels for the network.
            sz: The size of the convolutional kernel.
            conv_3D: A boolean flag indicating whether to use 3D convolution
                      layers (if True) or 2D convolution layers (if False).

        Returns:
            A sequential model containing the batch normalization layer
            and the convolutional layer.
    """
    conv_layer = nn.Conv3d if conv_3D else nn.Conv2d
    batch_norm = nn.BatchNorm3d if conv_3D else nn.BatchNorm2d
    return nn.Sequential(
        batch_norm(in_channels, eps=1e-5, momentum=0.05),
        conv_layer(in_channels, out_channels, sz, padding=sz // 2),
    )


class resdown(nn.Module):
    """
    A class representing a neural network model for performing convolutional operations.

    This class sets up convolutional layers and a projection layer, allowing for
    processing of input tensors through a specified architecture. It can be configured
    for 2D or 3D convolution based on the provided parameters.

    Methods:
        __init__: Initializes an instance of the class.
        forward: Computes the forward pass of the network.

    Attributes:
        layers: A sequential container for the convolutional layers.
        projection_layer: The projection layer used in the model.

    The __init__ method builds the architecture of the model, setting up the necessary
    convolutional layers and a projection layer based on the input specifications.
    The forward method processes an input tensor through these layers and returns
    the transformed output tensor.
    """

    def __init__(self, in_channels, out_channels, sz, conv_3D=False):
        """
        Initializes an instance of the class.

            This method sets up the convolutional layers and the projection layer
            for the neural network model. It creates a sequential container
            for the convolutional layers based on the specified input and
            output channels, kernel size, and whether 3D convolution is used.

            Args:
                in_channels: The number of input channels for the convolutional layers.
                out_channels: The number of output channels for the convolutional layers.
                sz: The size of the kernel to be used in the convolutional layers.
                conv_3D: A boolean flag indicating whether to use 3D convolutions.

            Returns:
                None
        """
        super().__init__()
        self.conv = nn.Sequential()
        self.proj = batchconv0(in_channels, out_channels, 1, conv_3D)
        for t in range(4):
            if t == 0:
                self.conv.add_module(
                    "conv_%d" % t, batchconv(in_channels, out_channels, sz, conv_3D)
                )
            else:
                self.conv.add_module(
                    "conv_%d" % t, batchconv(out_channels, out_channels, sz, conv_3D)
                )

    def forward(self, x):
        """
        Compute the forward pass of the network.

            This method takes an input tensor, processes it through the defined
            projection and convolution layers, and returns the transformed tensor.

            Args:
                x: The input tensor to be processed.

            Returns:
                The output tensor after applying projection and convolution operations.
        """
        x = self.proj(x) + self.conv[1](self.conv[0](x))
        x = x + self.conv[3](self.conv[2](x))
        return x


class downsample(nn.Module):
    """
    A class for downsampling operations in a neural network.

    This class sets up a sequential module for downsampling input data, utilizing either
    max pooling or average pooling based on the provided parameters. It includes functionality
    to create residual downsampling layers according to specified base sizes.

    Methods:
        __init__: Initializes the downsampling module.
        forward: Processes input data through downsampling layers.

    Attributes:
        nbase: A list of base sizes for the residual downsampling layers.
        sz: The size parameter used in the residual downsampling layers.
        conv_3D: Boolean indicating the use of 3D convolutions.
        max_pool: Boolean indicating the use of max pooling over average pooling.

    The __init__ method initializes the downsampling module with specified parameters, while
    the forward method processes the input tensor through the downsampling layers and
    returns a list of intermediate outputs.
    """

    def __init__(self, nbase, sz, conv_3D=False, max_pool=True):
        """
        Initializes the downsampling module of a neural network.

            This method sets up a sequential module for downsampling, either using
            max pooling or average pooling based on the parameters provided. It also
            creates a series of residual downsampling layers according to the base
            sizes specified.

            Args:
                nbase: A list of base sizes for the residual downsampling layers.
                sz: The size parameter used in the residual downsampling layers.
                conv_3D: A boolean indicating whether to use 3D convolutions.
                max_pool: A boolean indicating whether to use max pooling instead of
                    average pooling.

            Returns:
                None: This method initializes the object and does not return a value.
        """
        super().__init__()
        self.down = nn.Sequential()
        if max_pool:
            self.maxpool = (
                nn.MaxPool3d(2, stride=2) if conv_3D else nn.MaxPool2d(2, stride=2)
            )
        else:
            self.maxpool = (
                nn.AvgPool3d(2, stride=2) if conv_3D else nn.AvgPool2d(2, stride=2)
            )
        for n in range(len(nbase) - 1):
            self.down.add_module(
                "res_down_%d" % n, resdown(nbase[n], nbase[n + 1], sz, conv_3D)
            )

    def forward(self, x):
        """
        Processes the input data through a series of downsampling layers.

            This method takes an input tensor and sequentially applies a series of
            downsampling operations. Each operation reduces the spatial dimensions
            of the input, and the results are stored in a list that is returned at
            the end.

            Args:
                x: The input tensor to be processed through the downsampling layers.

            Returns:
                A list containing the outputs of each downsampling layer, which are
                the intermediate tensors after applying the respective transformations.
        """
        xd = []
        for n in range(len(self.down)):
            if n > 0:
                y = self.maxpool(xd[n - 1])
            else:
                y = x
            xd.append(self.down[n](y))
        return xd


class batchconvstyle(nn.Module):
    """
    A class that implements a neural network layer with batch convolution and style transfer capabilities.

    This class initializes a convolutional layer and a fully connected layer, allowing for
    the transformation of input tensors based on a specified style. It supports both 2D and
    3D convolutions and can optionally combine multiple input tensors.

    Methods:
        __init__
        forward

    Attributes:
        in_channels
        out_channels
        style_channels
        sz
        conv_3D

    The methods of this class include:
        - __init__: Sets up the neural network layers based on provided parameters,
          including input and output channel sizes, style channels, layer size,
          and whether to use 3D convolutions.
        - forward: Computes the forward pass by processing the input tensor
          and applying a transformation influenced by the style parameter,
          followed by a convolutional operation.

    The attributes manage the configuration of the layer including channel counts,
    size specifications, and whether to utilize 3D convolution.
    """

    def __init__(self, in_channels, out_channels, style_channels, sz, conv_3D=False):
        """
        Initializes the neural network layer.

            This method sets up the layers of the neural network by creating
            a batch convolution layer and a fully connected layer based on
            the provided parameters.

            Args:
                in_channels: The number of input channels for the convolution layer.
                out_channels: The number of output channels for the convolution layer.
                style_channels: The number of input features for the fully connected layer.
                sz: The size of the convolutional layer.
                conv_3D: A boolean indicating whether to use 3D convolution; defaults to False.

            Returns:
                None
        """
        super().__init__()
        self.concatenation = False
        self.conv = batchconv(in_channels, out_channels, sz, conv_3D)
        self.full = nn.Linear(style_channels, out_channels)

    def forward(self, style, x, mkldnn=False, y=None):
        """
        Compute the forward pass of the model.

            This method processes the input tensor `x` by optionally adding another input `y`,
            applies a transformation based on the specified style, and then passes the result through a convolutional layer.

            Args:
                style: The style parameter used to influence the transformation of the input tensor.
                x: The input tensor that will be processed.
                mkldnn: A boolean flag indicating whether to use MKL-DNN for optimized computations.
                y: An optional tensor to be added to `x`.

            Returns:
                The output tensor after applying the convolutional operation.
        """
        if y is not None:
            x = x + y
        feat = self.full(style)
        for k in range(len(x.shape[2:])):
            feat = feat.unsqueeze(-1)
        if mkldnn:
            x = x.to_dense()
            y = (x + feat).to_mkldnn()
        else:
            y = x + feat
        y = self.conv(y)
        return y


class resup(nn.Module):
    """
    A neural network module designed for processing input and style channels
    using convolutional layers, with support for optional 3D convolution.

    Methods:
        __init__: Initializes the neural network with specified parameters.
        forward: Computes the forward pass of the model.

    Attributes:
        in_channels: The number of input channels for the first convolution layer.
        out_channels: The number of output channels for the convolution layers.
        style_channels: The number of channels used for style processing in the network.
        sz: The size of the convolution filter.
        conv_3D: A boolean flag for enabling 3D convolutions.

    The methods and attributes facilitate the construction and functionality of
    a convolutional neural network tailored to process varying input and style
    data, optimizing performance through adjustable configurations.
    """

    def __init__(self, in_channels, out_channels, style_channels, sz, conv_3D=False):
        """
        Initializes a neural network module with convolutional layers.

            This constructor sets up a sequential model that includes multiple
            convolutional layers tailored for processing input channels and
            style channels, with an optional 3D convolution support.

            Args:
                in_channels: The number of input channels for the first convolution layer.
                out_channels: The number of output channels for the convolution layers.
                style_channels: The number of channels used for style processing in the network.
                sz: The size of the convolution filter.
                conv_3D: A boolean flag that indicates whether to use 3D convolutions (default is False).

            Returns:
                None
        """
        super().__init__()
        self.concatenation = False
        self.conv = nn.Sequential()
        self.conv.add_module(
            "conv_0", batchconv(in_channels, out_channels, sz, conv_3D=conv_3D)
        )
        self.conv.add_module(
            "conv_1",
            batchconvstyle(
                out_channels, out_channels, style_channels, sz, conv_3D=conv_3D
            ),
        )
        self.conv.add_module(
            "conv_2",
            batchconvstyle(
                out_channels, out_channels, style_channels, sz, conv_3D=conv_3D
            ),
        )
        self.conv.add_module(
            "conv_3",
            batchconvstyle(
                out_channels, out_channels, style_channels, sz, conv_3D=conv_3D
            ),
        )
        self.proj = batchconv0(in_channels, out_channels, 1, conv_3D=conv_3D)

    def forward(self, x, y, style, mkldnn=False):
        """
        Computes the forward pass of the model.

            This method takes input tensors and processes them through a series of
            convolutional layers and projections, returning the output after the
            computations.

            Args:
                x: The input tensor to be processed.
                y: An additional input tensor that may affect the output.
                style: A style tensor that impacts how the input is processed.
                mkldnn: A boolean flag indicating whether to use MKLDNN
                        for performance optimization.

            Returns:
                The output tensor after applying the forward pass operations.
        """
        x = self.proj(x) + self.conv[1](style, self.conv[0](x), y=y, mkldnn=mkldnn)
        x = x + self.conv[3](
            style, self.conv[2](style, x, mkldnn=mkldnn), mkldnn=mkldnn
        )
        return x


class make_style(nn.Module):
    """
    A class for processing input tensors to produce normalized style representations.

    This class is designed to set up the necessary layers for a model that
    utilizes either 2D or 3D average pooling to transform input data into a
    style representation.

    Methods:
        __init__: Initializes an instance of the class.
        forward: Processes the input tensor to produce a normalized style representation.

    Attributes:
        None

    The __init__ method sets up the model layers based on the specified pooling type,
    while the forward method takes an input tensor, applies average pooling,
    flattens the data, and normalizes it using the L2 norm to output a style representation.
    """

    def __init__(self, conv_3D=False):
        """
        Initializes an instance of the class.

            This constructor sets up the necessary layers for the model, including a
            flattening layer and an average pooling layer. The type of average pooling
            used is determined by the conv_3D parameter.

            Args:
                conv_3D: A boolean flag indicating whether to use 3D average pooling
                          (True) or 2D average pooling (False).

            Returns:
                None
        """
        super().__init__()
        self.flatten = nn.Flatten()
        self.avg_pool = F.avg_pool3d if conv_3D else F.avg_pool2d

    def forward(self, x0):
        """
        Processes the input tensor to produce a normalized style representation.

            This method takes a tensor as input, performs average pooling to reduce its
            dimensions, flattens the result, and normalizes it based on the L2 norm.

            Args:
                x0: The input tensor that contains the features to be processed.

            Returns:
                A tensor representing the normalized style, obtained from the input tensor.
        """
        style = self.avg_pool(x0, kernel_size=x0.shape[2:])
        style = self.flatten(style)
        style = style / torch.sum(style**2, axis=1, keepdim=True) ** 0.5
        return style


class upsample(nn.Module):
    """
    A class that implements an upsampling module consisting of residual upsampling layers.

    This module is designed to upscale input data by applying a series of transformation
    and upsampling layers. It can be configured for both 2D and 3D convolution operations,
    based on the needs of the application.

    Methods:
        __init__: Initializes the upsampling module with specified parameters.
        forward: Computes the forward pass of the model, transforming the input tensor.

    Attributes:
        nbase: A list of base channel sizes for each upsampling layer.
        sz: The size parameter used to configure the layers.
        conv_3D: A flag indicating whether to use 3D convolutions.

    The `__init__` method sets up the layers based on the base channel sizes, size parameter,
    and the convolution type (2D or 3D). The `forward` method processes the input tensor
    through the defined layers, applying the specified style for transformation and allows
    for MKLDNN optimization.
    """

    def __init__(self, nbase, sz, conv_3D=False):
        """
        Initialize the upsampling module.

            This method initializes an upsampling module that consists of a series
            of residual upsampling layers based on the provided base channels.

            Args:
                nbase: A list of base channel sizes for each layer to be created.
                sz: The size parameter used for configuring the layers.
                conv_3D: A boolean flag indicating whether to use 3D convolutions
                          instead of the standard layers; default is False.

            Returns:
                None
        """
        super().__init__()
        self.upsampling = nn.Upsample(scale_factor=2, mode="nearest")
        self.up = nn.Sequential()
        for n in range(1, len(nbase)):
            self.up.add_module(
                "res_up_%d" % (n - 1),
                resup(nbase[n], nbase[n - 1], nbase[-1], sz, conv_3D),
            )

    def forward(self, style, xd, mkldnn=False):
        """
        Computes the forward pass of the model.

            This method takes the input data and passes it through a series of upsampling
            and transformation layers defined in the model. It processes the input tensor
            by applying the specified style and handles the optional MKLDNN optimization.

            Args:
                style: The style information to be applied during the transformation.
                xd: A list of intermediate tensors to be used in the upsampling process.
                mkldnn: Optional flag to enable MKLDNN optimizations; defaults to False.

            Returns:
                The transformed output tensor after applying the style and upsampling.
        """
        x = self.up[-1](xd[-1], xd[-1], style, mkldnn=mkldnn)
        for n in range(len(self.up) - 2, -1, -1):
            if mkldnn:
                x = self.upsampling(x.to_dense()).to_mkldnn()
            else:
                x = self.upsampling(x)
            x = self.up[n](x, xd[n], style, mkldnn=mkldnn)
        return x


class CPnet(nn.Module):
    """
    CPnet is the Cellpose neural network model used for cell segmentation and image restoration.

    Args:
        nbase (list): List of integers representing the number of channels in each layer of the downsample path.
        nout (int): Number of output channels.
        sz (int): Size of the input image.
        mkldnn (bool, optional): Whether to use MKL-DNN acceleration. Defaults to False.
        conv_3D (bool, optional): Whether to use 3D convolution. Defaults to False.
        max_pool (bool, optional): Whether to use max pooling. Defaults to True.
        diam_mean (float, optional): Mean diameter of the cells. Defaults to 30.0.

    Attributes:
        nbase (list): List of integers representing the number of channels in each layer of the downsample path.
        nout (int): Number of output channels.
        sz (int): Size of the input image.
        residual_on (bool): Whether to use residual connections.
        style_on (bool): Whether to use style transfer.
        concatenation (bool): Whether to use concatenation.
        conv_3D (bool): Whether to use 3D convolution.
        mkldnn (bool): Whether to use MKL-DNN acceleration.
        downsample (nn.Module): Downsample blocks of the network.
        upsample (nn.Module): Upsample blocks of the network.
        make_style (nn.Module): Style module, avgpool's over all spatial positions.
        output (nn.Module): Output module - batchconv layer.
        diam_mean (nn.Parameter): Parameter representing the mean diameter to which the cells are rescaled to during training.
        diam_labels (nn.Parameter): Parameter representing the mean diameter of the cells in the training set (before rescaling).

    """

    def __init__(
        self,
        nbase,
        nout,
        sz,
        mkldnn=False,
        conv_3D=False,
        max_pool=True,
        diam_mean=30.0,
    ):
        """
        Initializes a neural network module with specified architecture parameters.

            This method sets up the network's architecture by initializing various
            components such as the downsampling, upsampling processes, and
            convolution layers based on the input parameters.

            Args:
                nbase: A list representing the base number of channels for the network.
                nout: The number of output channels for the final layer.
                sz: The size of the input to the network.
                mkldnn: A boolean indicating whether to use MKL-DNN for optimized operations.
                conv_3D: A boolean indicating whether to use 3D convolutions.
                max_pool: A boolean indicating whether to use max pooling in the downsample process.
                diam_mean: A float representing the mean diameter parameter for the network.

            Returns:
                None
        """
        super().__init__()
        self.nchan = nbase[0]
        self.nbase = nbase
        self.nout = nout
        self.sz = sz
        self.residual_on = True
        self.style_on = True
        self.concatenation = False
        self.conv_3D = conv_3D
        self.mkldnn = mkldnn if mkldnn is not None else False
        self.downsample = downsample(nbase, sz, conv_3D=conv_3D, max_pool=max_pool)
        nbaseup = nbase[1:]
        nbaseup.append(nbaseup[-1])
        self.upsample = upsample(nbaseup, sz, conv_3D=conv_3D)
        self.make_style = make_style(conv_3D=conv_3D)
        self.output = batchconv(nbaseup[0], nout, 1, conv_3D=conv_3D)
        self.diam_mean = nn.Parameter(
            data=torch.ones(1) * diam_mean, requires_grad=False
        )
        self.diam_labels = nn.Parameter(
            data=torch.ones(1) * diam_mean, requires_grad=False
        )

    @property
    def device(self):
        """
        Get the device of the model.

        Returns:
            torch.device: The device of the model.
        """
        return next(self.parameters()).device

    def forward(self, data):
        """
        Forward pass of the CPnet model.

        Args:
            data (torch.Tensor): Input data.

        Returns:
            tuple: A tuple containing the output tensor, style tensor, and downsampled tensors.
        """
        if self.mkldnn:
            data = data.to_mkldnn()
        T0 = self.downsample(data)
        if self.mkldnn:
            style = self.make_style(T0[-1].to_dense())
        else:
            style = self.make_style(T0[-1])
        style0 = style
        if not self.style_on:
            style = style * 0
        T1 = self.upsample(style, T0, self.mkldnn)
        T1 = self.output(T1)
        if self.mkldnn:
            T0 = [t0.to_dense() for t0 in T0]
            T1 = T1.to_dense()
        return T1, style0, T0

    def save_model(self, filename):
        """
        Save the model to a file.

        Args:
            filename (str): The path to the file where the model will be saved.
        """
        torch.save(self.state_dict(), filename)

    def load_model(self, filename, device=None):
        """
        Load the model from a file.

        Args:
            filename (str): The path to the file where the model is saved.
            device (torch.device, optional): The device to load the model on. Defaults to None.
        """
        if (device is not None) and (device.type != "cpu"):
            state_dict = torch.load(filename, map_location=device, weights_only=True)
        else:
            self.__init__(
                self.nbase,
                self.nout,
                self.sz,
                self.mkldnn,
                self.conv_3D,
                self.diam_mean,
            )
            state_dict = torch.load(
                filename, map_location=torch.device("cpu"), weights_only=True
            )

        if state_dict["output.2.weight"].shape[0] != self.nout:
            for name in self.state_dict():
                if "output" not in name:
                    self.state_dict()[name].copy_(state_dict[name])
        else:
            self.load_state_dict(
                dict([(name, param) for name, param in state_dict.items()]),
                strict=False,
            )


class CPnetBioImageIO(CPnet):
    """
    A subclass of the CPnet model compatible with the BioImage.IO Spec.

    This subclass addresses the limitation of CPnet's incompatibility with the BioImage.IO Spec,
    allowing the CPnet model to use the weights uploaded to the BioImage.IO Model Zoo.
    """

    def forward(self, x):
        """
        Perform a forward pass of the CPnet model and return unpacked tensors.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: A tuple containing the output tensor, style tensor, and downsampled tensors.
        """
        output_tensor, style_tensor, downsampled_tensors = super().forward(x)
        return output_tensor, style_tensor, *downsampled_tensors

    def load_model(self, filename, device=None):
        """
        Load the model from a file.

        Args:
            filename (str): The path to the file where the model is saved.
            device (torch.device, optional): The device to load the model on. Defaults to None.
        """
        if (device is not None) and (device.type != "cpu"):
            state_dict = torch.load(filename, map_location=device, weights_only=True)
        else:
            self.__init__(
                self.nbase,
                self.nout,
                self.sz,
                self.mkldnn,
                self.conv_3D,
                self.diam_mean,
            )
            state_dict = torch.load(
                filename, map_location=torch.device("cpu"), weights_only=True
            )

        self.load_state_dict(state_dict)

    def load_state_dict(self, state_dict):
        """
        Load the state dictionary into the model.

        This method overrides the default `load_state_dict` to handle Cellpose's custom
        loading mechanism and ensures compatibility with BioImage.IO Core.

        Args:
            state_dict (Mapping[str, Any]): A state dictionary to load into the model
        """
        if state_dict["output.2.weight"].shape[0] != self.nout:
            for name in self.state_dict():
                if "output" not in name:
                    self.state_dict()[name].copy_(state_dict[name])
        else:
            super().load_state_dict(
                {name: param for name, param in state_dict.items()}, strict=False
            )
