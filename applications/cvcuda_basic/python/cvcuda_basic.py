"""
SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""  # no qa
import warnings
from collections.abc import Sequence

import cupy as cp
import os
from argparse import ArgumentParser

import cvcuda
import nvcv
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import HolovizOp, VideoStreamReplayerOp


flip_code_dict = {"horizontal": 1, "vertical": 0, "both": -1, "none": 2}

border_code_dict = {
    "constant": cvcuda.Border.CONSTANT,
    "reflect": cvcuda.Border.REFLECT,
    "reflect101": cvcuda.Border.REFLECT101,
    "replicate": cvcuda.Border.REPLICATE,
    "wrap": cvcuda.Border.WRAP,
}


class ChannelTensorOp(Operator):
    def __init__(
        self,
        fragment,
        *args,
        out_tensor_name="scale",
        out_port_name="scale",
        values=(1, 1, 1),
        dtype=cp.float32,
        **kwargs,
    ):
        self.dtype = dtype
        self.values = values
        self.out_tensor_name = out_tensor_name
        self.out_port_name = out_port_name
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output(self.out_port_name)

    def compute(self, op_input, op_output, context):
        tensor = cp.asarray(self.values, dtype=self.dtype)
        tensor = tensor.reshape(1, 1, 1, len(self.values))
        op_output.emit({self.out_tensor_name: tensor}, self.out_port_name)


# Define custom Operators for use in the demo
class CropFlipNormalizeReformatOp(Operator):
    """Example of an operator processing input video (as a tensor).

     This operator has:
         inputs:  "input_tensor"
         outputs: "output_tensor"

    Each input frame is processed by CV-CUDA's crop_flip_normalize_format operator to perform
    multiple pre-processing steps using a single kernel.

    This operation performs the following steps with a single, fused CUDA kernel:

        - Pad and Crop the input image to the specified rectangle.
        - Flip the cropped image horizontally and/or vertically (or neither).
        - Normalize the flipped image using the provided ``base`` and ``scale``.
        - Convert the normalized image to the specified output data type.
        - Reformat the normalized image to the specified output layout.

    The scaling and offset supplied via `scale` or `base` are specified per-channel, while gobal_scale
    and global_shift are applied to all channels. The overall scaling is applied as::

        output_tensor = (input_tensor - base) * scale * global_scale + global_shift

    In the documentation below H = height, W = width, C=channels, N=batch size.

    Valid input tensors for this operator are 2D (HW), 3D (HWC or CHW) or 4D (NHWC or NCHW).
    The output tensor can have any of the same formats, but if ``out_tensor_shape`` is specified,
    it must match the input in both the batch size (N) and number of channels (C). If those axes
    don't exist on the input, but should be present in the output, they can be set to size 1.

    Device memory for the output tensor as well as small internal tensors for ``flip``, ``base``,
    ``rect`` and ``scale`` will be allocated on the first call to compute. Subsequent calls will
    reuse these tensors so that the operator can be used efficiently in a streaming pipeline.

    Parameters
    ----------
    flip : {"horizontal", "vertical", "both", "none", None}, optional
        Flip the input tensor along the specified direction(s). If ``None`` or "none", no flipping
        is performed.
    in_tensor_format : {"HW", "HWC", "CHW", "NHWC", "NCHW"}, optional
        The format of the input tensor where H = height, W = width, C = channels, N = batch. If
        ``None``, the following assumption will be made based on the number of input dimensions:

            - 2D: "HW"
            - 3D: "HWC"
            - 4D: "NHWC"

    out_tensor_format : {"HW", "HWC", "CHW", "NHWC", "NCHW"}, optional
        The format of the input tensor where H = height, W = width, C = channels, N = batch.
        If ``None``, it will be set equal to ``in_tensor_format``.
    out_tensor_shape : tuple of int, optional
        If a tuple is provided, its length must equal the number of dimensions specified for
        ``out_tensor_format``. For the default value of None:

            - if "N" is in ``out_tensor_format`` it will match N for the input tensor
            - if "C" is in ``out_tensor_format`` it will match C for the input tensor
            - W will be rect[2] if rect is provided, otherwise it will match the input tensor
            - H will be rect[3] if rect is provided, otherwise it will match the input tensor

    out_tensor_dtype : {"int8", "int16", "int32", "uint8", "uint16", "uint32", "float32"}, optional
        The data type of the output tensor. If None, it will be set to the input tensor dtype.
        Can also be provided as any type that cupy.dtype would cast to one of the support types.

    rect : tuple of int or None, default=None
        The crop rectangle in the format ``(crop_x, crop_y, crop_width_x, crop_width_y)``. The
        coordinate conventions are as shown in the diagram below (x runs left to right, while
        y runs top to bottom). A negative value for ``crop_x`` or ``crop_y`` will result in
        padding the image. Similarly, using ``crop_width_x`` or `crop_width_y`` that are larger
        than the input tensor will result in padding. If None, no cropping or padding is
        performed. Coordinate conventions::

        ```
                                                                               x
            ----------------------------------------------------------->
            |
            |   (crop_x, crop_y)
            |           ____________________________________________
            |          |                                            |
            |          |                                            |
            |          |                                            |
            |          |                                            |
            |          |                                            |
            |          |____________________________________________|
            |
            |                                  (crop_x + crop_width_x, crop_y + crop_width_y)
            |
          y v
        ```

    base : 4-tuple of float, array-like of size 4, str or None
        The base values to use for normalization. If a tuple or tensor is provided, it should
        have 1 value per channel present in the input tensor. If instead the `base` value
        should be obtained from a received tensor, then instead specify a string for this
        argument in the form "port_name.tensor_name" to specifying the name of the tensor and the
        name of the input port where it can be found. The specified input port will be created.
        If just the tensor name is provided, that tensor will be searched for on the existing
        "in_image" input port instead. If None (default), all base values will be set to 0.0.
    scale : 4-tuple of float, array-like of size 4 or str
        The scale values to use for normalization. If a tuple or tensor is provided, it should
        have 1 value per channel present in the input tensor. If instead the `base` value
        should be obtained from a received tensor, then instead specify a string for this
        argument in the form "port_name.tensor_name" to specifying the name of the tensor and the
        name of the input port where it can be found. The specified input port will be created.
        If just the tensor name is provided, that tensor will be searched for on the existing
        "in_image" input port instead. If None (default), all scale values will be set to 1.0.
    border: {"constant", "reflect", "reflect101", "replicate", "wrap"} or cvcuda.Border enum, optional
        The boundary extension mode to use if ``rect`` results in padding. The default is
        "constant". Can also be provided
    bvalue: int, optional
        The value to use for padding if ``border`` is set to "constant". The default is 0.
    global_scale: float, optional
        A global scaling factor to apply to all channels after normalization. The default is 1.0.
    global_shift: float, optional
        A global shift factor to apply to all channels after normalization. The default is 0.0.
    flags : 0 or cvcuda.NormalizeFlags.SCALE_IS_STDDEV, optional
        If set to ``cvcuda.NormalizeFlags.SCALE_IS_STDDEV``. In this case, `scale` is treated as a
        standard deviation and the scaling applied along axis, i, is
        ``1 / (scale[i] * scale[i] + epsilon)`` instead of ``scale[i]``. If 0, ``epsilon`` is unused.
    epsilon : float or None,
        An epsilon value used to avoid potential divide by zero during scaling when ``flags`` is set to
        ``cvcuda.NormalizeFlags.SCALE_IS_STDDEV``. Unused otherwise.
    stream : cvcuda.Stream or None, optional
        The CUDA stream to use for the operation. If None, the default stream will be used.
    """  # noqa: E501

    def __init__(
        self,
        fragment,
        *args,
        flip=None,
        in_tensor_format=None,
        out_tensor_format=None,
        out_tensor_shape=None,
        out_tensor_dtype=None,
        rect=None,
        base=None,
        scale=None,
        border="constant",
        bvalue=0,
        global_scale=1.0,
        global_shift=0.0,
        flags=0,
        epsilon=None,
        stream=None,
        **kwargs,
    ):
        # name of the input port
        self.in_image_port = "in_image"

        # convert flip to integer code stored on device
        self._set_flip_code(flip)

        # validate and set self.border and self.bvalue
        self._prep_border_attributes(border, bvalue)

        # validate and set self.rect and self.rect_width_height
        self._prep_rect_attributes(rect)

        # Validate and set self.in_tensor_format
        # If None, self.in_tensor_format will be determined later via the input tensor shape in the
        # compute method.
        #
        # If not None, also sets self.in_index_n, self.in_index_c, self.in_index_h, self.in_index_w
        # which are integers indicating which input axis if any corresponds to dimensions
        # N (batch), C (channels), H (height) or W (width).
        self._prep_in_tensor_attributes(in_tensor_format)

        # validate and set self.out_tensor_format, self.out_tensor_shape, self.out_tensor_dtype,
        # and self.out_tensor_layout.
        self._prep_out_tensor_attributes(out_tensor_format, out_tensor_shape, out_tensor_dtype)

        # Helper methods handle base and scale tensors which can be provided as either:
        # - a tuple of values (1 per channel)
        # - a tensor-like object (cp.ndarray, cvcuda.Tensor)
        # - a "name" string specifying the name of a tensor present on the in_image input port
        # - a "port.name" string specifying the name of a port and the tensor to be found on it
        # self.rect, self.rect_tensor_name, self.rect_tensor_port = self._prep_per_channel_tensor(
        #     rect,
        #     default_value=None,
        # )
        self.base, self.base_tensor_name, self.base_tensor_port = self._prep_per_channel_tensor(
            base,
            default_value=0.0,
        )
        self.scale, self.scale_tensor_name, self.scale_tensor_port = self._prep_per_channel_tensor(
            scale,
            default_value=1.0,
        )

        # global scale and shift
        self.global_scale = global_scale
        self.global_shift = global_shift

        # validate and set self.flags and self.epsilon
        self._prep_flags_and_epsilon(flags, epsilon)

        # CUDA stream
        self.stream = stream

        # output CuPy array (to be set in compute method)
        # This inernal array will always have self.out_tensor_layout format of NHWC or NCHW, but
        # the singleton N or C dimension will be dropped from the emitted tensor if
        # self.out_tensor_format is "HW" or "HWC" or "CHW".
        self.cupy_out = None
        # output cvcuda.Tensor view of cupy_out (shares same memory pointer)
        self.cv_out = None

        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def _set_flip_code(self, flip):
        """Store flip code as an int32 NC tensor of shape (1, 1) on the device"""
        if flip is None:
            flip_code = 2
        elif isinstance(flip, int):
            flip_code = flip
        elif isinstance(flip, str):
            flip = flip.lower()
            if flip not in flip_code_dict:
                raise ValueError(f"Invalid flip value. Must be one of {tuple(flip_code_dict.keys())}")
            flip_code = flip_code_dict[flip]
        flip_code_tensor = cp.full((1, 1), flip_code, dtype=cp.int32)
        self.flip_code = cvcuda.as_tensor(flip_code_tensor, layout="NC")

    def _prep_border_attributes(self, border, bvalue):
        """set cvcuda.Border enum value and bvalue to be used in the constant extension case"""
        # convert border name to cvcuda.Border type
        if isinstance(border, cvcuda.Border):
            self.border = border
        elif not isinstance(border, str) or border.lower() not in border_code_dict:
            raise ValueError(f"Invalid flip value. Must be one of {tuple(border_code_dict.keys())}")
        self.border = border_code_dict[border]

        # set border value
        if not cp.isscalar(bvalue):
            raise ValueError("bvalue must be a scalar value")
        self.bvalue = bvalue

    def _prep_rect_attributes(self, rect):
        """Prepare attributes related to the rect argument

        If rect is not None this function:

            - converts to an int32 tensor of shape (1, 1, 1, 4) in NHWC format.
            - sets self.rect_height_width as (crop_width_y, crop_width_x)
        """
        rect_tensor_name = None
        rect_tensor_port = None
        if rect is not None:
            if isinstance(rect, str):
                n_period = rect.count(".")
                if n_period == 0:
                    rect_tensor_name = rect
                    rect_tensor_port = self.in_image_port
                elif n_period == 1:
                    rect_tensor_port, rect_tensor_name = rect.split(".")
                else:
                    raise ValueError("tensor name must be in the form 'name' or 'port.name'")
            elif isinstance(rect, tuple):
                if len(rect) != 4:
                    raise ValueError(f"Invalid rect tuple: {rect}. It must be a 4-tuple.")
                if not all(isinstance(i, int) for i in rect):
                    raise ValueError("rect tuple must contain integers.")
                self.rect_height_width = (rect[3], rect[2])
                num_batch = 1  # tuple input assumes batch size 1
                rect = cp.asarray(rect, dtype=cp.int32).reshape((num_batch, 1, 1, 4))
                rect = cvcuda.as_tensor(rect, layout="NHWC")
            else:
                if not isinstance(rect, cvcuda.Tensor):
                    rect = cvcuda.as_tensor(rect, "NHWC")
                if not rect.shape != (1, 1, 1, 4):
                    raise ValueError(f"Invalid rect shape: {rect.shape}. It must be (1, 1, 1, 4).")
                if rect.dtype.kind not in ["i", "u"]:
                    raise ValueError(f"rect should have an integer data type, not {rect.dtype.name}")
                tmp = cp.asnumpy(cp.asarray(rect.cuda())[0, 0, 0, :])
                self.rect_height_width = (tmp[3], tmp[2])
        else:
            self.rect_height_width = None  # no explicitly provided width, height via crop rectangle
        self.rect = rect
        self.rect_tensor_port = rect_tensor_port
        self.rect_tensor_name = rect_tensor_name

    def _set_in_axis_indices(self, in_tensor_format):
        """Determine which array axes corresponds to each dimension (N, H, W, C).

        Any axes that are not present in the format string will be set to -1.
        """
        self.in_index_n = in_tensor_format.find("N")
        self.in_index_c = in_tensor_format.find("C")
        self.in_index_h = in_tensor_format.find("H")
        self.in_index_w = in_tensor_format.find("W")

    def _prep_in_tensor_attributes(self, in_tensor_format):
        """Validate the input tensor format and set the input axis indices attributes"""
        if in_tensor_format is not None:
            in_tensor_format = in_tensor_format.upper()
            if in_tensor_format not in ["HW", "HWC", "CHW", "NHWC", "NCHW"]:
                raise ValueError(
                    f"Invalid out_layout value {in_tensor_format}. Must be one of ('HW', 'HWC', "
                    "'CHW', 'NHWC', 'NCHW')"
                )
            # determine which axis correspondings to each value of N, H, W, C (-1 for any not present)
            self._set_in_axis_indices(in_tensor_format)
        self.in_tensor_format = in_tensor_format

    def _prep_out_tensor_attributes(self, out_tensor_format, out_tensor_shape, out_tensor_dtype):
        """Validate the output tensor shape, dtype and format if provided."""
        # check for valid output format
        out_tensor_format = out_tensor_format.upper()
        if out_tensor_format not in ["HW", "HWC", "CHW", "NHWC", "NCHW"]:
            raise ValueError(
                f"Invalid out_tensor_format value {out_tensor_format}. Must be one of ('NHWC', "
                "'NCHW')."
            )

        ndim_out = len(out_tensor_format)
        self.out_tensor_format = out_tensor_format
        if out_tensor_format in ["CHW", "NCHW"]:
            self.out_tensor_layout = "NCHW"
        else:
            self.out_tensor_layout = "NHWC"

        # check that output shape is valid given the specified format
        if out_tensor_shape is not None:
            if (
                not isinstance(out_tensor_shape, Sequence)
                or len(out_tensor_shape) != ndim_out
                or not all(isinstance(i, int) for i in out_tensor_shape)
            ):
                raise ValueError(
                    f"Invalid out_tensor_shape: {out_tensor_shape} for format {out_tensor_format}."
                    " Must be a {ndim_out}-tuple corresponding to a NHWC or NCHW layout."
                )
            out_tensor_shape = tuple(out_tensor_shape)
            if out_tensor_shape[-1] not in [1, 3, 4]:
                raise ValueError(
                    f"Invalid out_tensor_shape: {out_tensor_shape}. Only 1, 3 or 4 channels are "
                    "supported."
                )
            # have to add a batch dimension for use with `crop_flip_normalize_reformat_into``
            # (the batch dimension will be removed again before emitting the tensor)
            if ndim_out == 2:
                if self.out_tensor_layout == "NHWC":
                    out_tensor_shape = (1,) + out_tensor_shape + (1,)
                else:
                    out_tensor_shape = (1, 1) + out_tensor_shape
            elif ndim_out == 3:
                out_tensor_shape = (1,) + out_tensor_shape

        # allow out_shape = None to mean match the input shape
        self.out_tensor_shape = out_tensor_shape

        # check that the specified output dtype is supported
        if out_tensor_dtype is not None:
            # Note: cupy.dtype is als numpy.dtype
            out_tensor_dtype = cp.dtype(out_tensor_dtype)
            if (
                out_tensor_dtype.kind not in ["u", "i", "f"]
                or (out_tensor_dtype.kind == "f" and out_tensor_dtype.itemsize != 4)
                or (
                    out_tensor_dtype.kind in ["u", "i"]
                    and out_tensor_dtype.itemsize < 1
                    or out_tensor_dtype.itemsize > 4
                )
            ):
                raise ValueError(
                    "out_tensor_dtype must be one of ('int8', 'int16', 'int32', 'uint8', "
                    "'uint16', 'uint32' or 'float32')"
                )
        self.out_tensor_dtype = out_tensor_dtype

    def _prep_per_channel_tensor(self, tensor, default_value=1.0):
        """Set tensor-like input as a cvcuda.Tensor of shape (1, 1, 1, num_channels) in NHWC format

        or specify a tensor name and-or port name to be used to find the tensor in the input.

        This is a common helper method used by both _prep_base_tensor_attribues and
        _prep_scale_tensor_attribues.
        """
        tensor_name = None
        tensor_port = None
        out_tensor = None
        if isinstance(tensor, str):
            n_period = tensor.count(".")
            if n_period == 0:
                tensor_name = tensor
                tensor_port = self.in_image_port
            elif n_period == 1:
                tensor_port, tensor_name = tensor.split(".")
            else:
                raise ValueError("tensor name must be in the form 'name' or 'port.name'")
        elif tensor is not None:
            if isinstance(tensor, tuple):
                if len(tensor) not in [1, 3, 4]:
                    raise ValueError("len(tensor) must match the number of channels")
                tensor = cp.asarray(tensor, dtype=cp.float32)
                tensor = tensor.reshape(1, 1, 1, len(tensor))
                out_tensor = cvcuda.as_tensor(tensor, layout="NHWC")
            elif isinstance(tensor, cvcuda.Tensor):
                out_tensor = tensor
            elif isinstance(tensor, cp.ndarray):
                if tensor.size not in [1, 3, 4]:
                    raise ValueError("tensor.size must match the number of channels")
                tensor = tensor.reshape(1, 1, 1, tensor.size)
                out_tensor = cvcuda.as_tensor(tensor, layout="NHWC")
            else:
                out_tensor = cvcuda.as_tensor(tensor, layout="NHWC")
        else:
            out_tensor = cvcuda.as_tensor(
                cp.full((1, 1, 1, 4), default_value, dtype=cp.float32), layout="NHWC"
            )
        return out_tensor, tensor_name, tensor_port

    def _prep_flags_and_epsilon(self, flags, epsilon):
        if flags != 0 and flags != cvcuda.NormalizeFlags.SCALE_IS_STDDEV:
            raise ValueError("The only supported flag is cvcuda.NormalizeFlags.SCALE_IS_STDDEV")
        self.flags = flags

        if flags == 0 and epsilon != 0.0:
            warnings.warn(
                "Provided value of epsilon will not be used unless flags is set to "
                "cvcuda.NormalizeFlags.SCALE_IS_STDDEV"
            )
        if epsilon is None:
            epsilon = 0.0
        self.epsilon = epsilon

    def setup(self, spec: OperatorSpec):
        spec.input(self.in_image_port)
        if (self.rect_tensor_name is not None) and (self.rect_tensor_port != self.in_image_port):
            spec.input(self.rect_tensor_port)
        if (self.base_tensor_port is not None) and (self.base_tensor_port != self.in_image_port):
            spec.input(self.base_tensor_port)
        if (self.scale_tensor_port is not None) and (self.scale_tensor_port != self.in_image_port):
            spec.input(self.scale_tensor_port)
        spec.output("out_image")

    def compute(self, op_input, op_output, context):
        tensormap = op_input.receive(self.in_image_port)
        input_tensor = tensormap[""]  # stride (2562, 3, 1)

        # TODO: instead of this assumption, provide in_tensor_format argument
        if self.in_tensor_format is None:
            if input_tensor.ndim == 2:
                self.in_tensor_format = "HW"
            elif input_tensor.ndim == 3:
                self.in_tensor_format = "HWC"
            elif input_tensor.ndim == 4:
                self.in_tensor_format = "NHWC"
            else:
                raise ValueError("Input tensor should have HW, HWC or NHWC dimensions")
            self._set_in_axis_indices(self.in_tensor_format)

        # determine number of channels from input tensor
        num_channels = 1 if self.in_index_c == -1 else input_tensor.shape[self.in_index_c]
        # determine batch size from the input tensor
        num_batch = 1 if self.in_index_n == -1 else input_tensor.shape[self.in_index_n]
        # print(f"{self.in_index_n = }")
        # print(f"{self.in_index_h = }")
        # print(f"{self.in_index_w = }")
        # print(f"{self.in_index_c = }")
        # print(f"{num_channels = }")
        # print(f"{num_batch = }")

        cv_in_tensor = nvcv.as_tensor(input_tensor, layout=self.in_tensor_format)
        cv_in = cvcuda.as_image(cv_in_tensor.cuda())

        # crop_flip_normalize_reformat only accepts a variable-shape image batch as input
        num_images = 1
        cv_img_batch = cvcuda.ImageBatchVarShape(num_images)  # TODO: can in_tensor have num_batch > 1?
        cv_img_batch.pushback(cv_in)

        if self.rect_tensor_name is not None:
            if self.rect_tensor_port == self.in_image_port:
                rect_tensormap = tensormap
            else:
                rect_tensormap = op_input.receive(self.rect_tensor_port)
            if self.rect_tensor_name not in rect_tensormap:
                raise ValueError(f"Could not find rect tensor with name {self.rect_tensor_name}")
            rect_tensor = rect_tensormap[self.rect_tensor_name]
            if isinstance(rect_tensor, tuple):
                if len(tuple) != 4 or not all(isinstance(i, int) for i in rect_tensor):
                    raise ValueError("rect tuple must contain 4 integers")
                self.rect_height_width = (rect_tensor[3], rect_tensor[2])
                self.rect = cp.asarray(rect_tensor, dtype=cp.int32).reshape((1, 1, 1, 4))
            elif isinstance(rect_tensor, cvcuda.Tensor):
                self.rect = rect_tensor
                if self.out_tensor_shape is None:
                    # assumes crop_width_x and crop_width_y don't change between frames
                    rect_tensor_cupy = cp.asarray(rect_tensor.cuda())
                    if rect_tensor_cupy.size != 4:
                        raise ValueError("tensor received on rect port should have 4 elements")
                    host_rect = cp.asnumpy(rect_tensor_cupy.ravel())
                    self.rect_height_width = (host_rect[3], host_rect[2])
            else:
                rect_tensor = cp.asarray(rect_tensor, dtype=cp.int32)
                if rect_tensor.size != 4:
                    raise ValueError("tensor received on rect port should have 4 elements")
                if self.out_tensor_shape is None:
                    # assumes crop_width_x and crop_width_y don't change between frames
                    host_rect = cp.asnumpy(rect_tensor[0, 0, 0, :])
                    self.rect_height_width = (host_rect[3], host_rect[2])
                self.rect = cvcuda.as_tensor(rect_tensor.reshape(1, 1, 1, 4), layout="NHWC")
        elif self.rect is None:
            # default is no cropping or padding
            # TODO: wrong integer type here can result in silently causing all zeros output!
            rect = cp.zeros((1, 1, 1, 4), dtype=cp.int32)
            # values are [crop_x, crop_y, crop_width_x, crop_width_y]
            rect[0, 0, 0, :] = cp.array(
                [0, 0, input_tensor.shape[self.in_index_w], input_tensor.shape[self.in_index_h]]
            )
            self.rect_height_width = (input_tensor.shape[self.in_index_h], input_tensor.shape[self.in_index_w])
            self.rect = cvcuda.as_tensor(rect, layout="NHWC")

        if self.base_tensor_port is not None:
            if self.base_tensor_port == self.in_image_port:
                base_tensormap = tensormap
            else:
                base_tensormap = op_input.receive(self.base_tensor_port)
            if self.base_tensor_name not in base_tensormap:
                raise ValueError(f"Could not find base tensor with name {self.base_tensor_name}")
            self.base = cvcuda.as_tensor(base_tensormap[self.base_tensor_name], layout="NHWC")

        if self.scale_tensor_port is not None:
            if self.base_tensor_port == self.in_image_port:
                scale_tensormap = tensormap
            else:
                scale_tensormap = op_input.receive(self.scale_tensor_port)
            if self.scale_tensor_name not in scale_tensormap:
                raise ValueError(f"Could not find scale tensor with name {self.scale_tensor_name}")
            self.scale = cvcuda.as_tensor(scale_tensormap[self.scale_tensor_name], layout="NHWC")

        # allocate the output tensor
        if self.cv_out is None:
            if self.out_tensor_shape is None:
                # determine output height and width from input tensor or specified rect argument
                if self.rect_height_width is not None:
                    height_width = self.rect_height_width
                else:
                    height_width = (
                        input_tensor.shape[self.in_index_h],
                        input_tensor.shape[self.in_index_w],
                    )

                # output tensor with same num_batch and num_channels as the input
                if self.out_tensor_layout == "NHWC":
                    out_shape = (num_batch, height_width[0], height_width[1], num_channels)
                else:  # NCHW
                    out_shape = (num_batch, num_channels, height_width[0], height_width[1])
            else:
                # _prep_out_tensor_attributes will have already assured len(out_shape) == 4
                out_shape = self.out_tensor_shape
                out_channels = out_shape[-1] if self.out_tensor_layout == "NHWC" else out_shape[1]
                if out_channels != num_channels:
                    raise ValueError(
                        f"Number of output channels ({out_channels}) must match the number of input "
                        f"channels ({num_channels})"
                    )
                if out_shape[0] != num_batch:
                    raise ValueError(
                        f"Number of images in batch ({out_shape[0]}) must match the number of input "
                        f"images ({num_batch})"
                    )

            if self.out_tensor_dtype is None:
                # cast to CuPy array to get the CuPy dtype (input_tensor.dtype is the DLPack DLDataType)
                self.out_tensor_dtype = cp.asarray(input_tensor).dtype
            # print(f"{out_shape = }")
            # print(f"{self.rect_height_width = }")
            # print(f"{num_batch = }")
            # print(f"{num_channels = }")
            # print(f"{self.out_tensor_format = }")

            # cv_out will share cupy_out's memory so assign it to the class so it stays alive
            self.cupy_out = cp.empty(out_shape, dtype=self.out_tensor_dtype)
            self.cv_out = cvcuda.as_tensor(self.cupy_out, layout="NHWC")

        # use the already-allocated output buffer (self.cv_out)
        cvcuda.crop_flip_normalize_reformat_into(
            dst=self.cv_out,
            src=cv_img_batch,
            rect=self.rect,
            flip_code=self.flip_code,
            base=self.base,
            scale=self.scale,
            # arguments below this point are optional
            globalscale=self.global_scale,
            globalshift=self.global_shift,
            flags=self.flags,
            epsilon=self.epsilon,
            border=self.border,
            bvalue=self.bvalue,
            stream=self.stream,
        )

        # drop N and/or C dimensions to match the specified output format
        if self.out_tensor_format == "HW":
            if self.out_tensor_layout == "NHWC":
                out_cupy = self.cupy_out[0, ..., 0]
            else:  # NCHW
                out_cupy = self.cupy_out[0, 0, ...]
        elif self.out_tensor_format in ["HWC", "CHW"]:
            out_cupy = self.cupy_out[0, ...]
        else:
            out_cupy = self.cupy_out

        output_tensormap = dict(image=out_cupy)
        op_output.emit(output_tensormap, "out_image")


# Now define a simple application using the operators defined above
class MyVideoProcessingApp(Application):
    """Example of an application that uses the operators defined above.

    This application has the following operators:

    - VideoStreamReplayerOp
    - CropFlipNormalizeReformatOp
    - HolovizOp

    The VideoStreamReplayerOp reads a video file and sends the frames to the CropFlipNormalizeReformatOp.
    The CropFlipNormalizeReformatOp processes the frames and sends the processed frames to the HolovizOp.
    The HolovizOp displays the processed frames.
    """

    def __init__(self, *args, data, count=0, **kwargs):
        super().__init__(*args, **kwargs)

        if data == "none":
            data = os.path.join(os.environ.get("HOLOHUB_DATA_PATH", "../data"), "endoscopy")

        self.sample_data_path = data
        self.count = count

    def compose(self):
        width = 854
        height = 480
        video_dir = self.sample_data_path
        if not os.path.exists(video_dir):
            raise ValueError(f"Could not find video data: {video_dir=}")
        source = VideoStreamReplayerOp(
            self,
            name="replayer",
            directory=video_dir,
            basename="surgical_video",
            frame_rate=0,
            repeat=True,
            realtime=True,
            count=self.count,
        )

        # (crop_start_x, crop_start_y, crop_width_x, crop_width_y)

        rect = (-120, 120, 854+240, 480 - 2*120)
        # rect = None
        use_external_rect_tensor = False
        use_external_scale_tensor = False
        use_external_base_tensor = False
        # rect = None
        image_processing = CropFlipNormalizeReformatOp(
            self,
            flip="none",
            name="image_processing",
            rect="rect.rect" if use_external_rect_tensor else rect,
            border="constant",
            bvalue=0,
            global_scale=1.4,
            global_shift=0.0,
            base="base.base" if use_external_base_tensor else (0.0, 0.0, 0.0),
            scale="scale.scale" if use_external_scale_tensor else (1.0, 1.0, 1.0),
            # out_tensor_shape=(480, 854, 3),
            out_tensor_format="HWC",
            # out_tensor_dtype=cp.float32,
        )

        if rect is None:
            height_viz = height
            width_viz = width
        else:
            height_viz = rect[3]  # (rect[3] // 32) * 32
            width_viz = rect[2]  # (rect[2] // 32) * 32

        if use_external_rect_tensor:
            if rect is None:
                raise ValueError("must specifiy rect if using external rect tensor")
            rect_tensor_op = ChannelTensorOp(
                self,
                out_port_name="rect",
                out_tensor_name="rect",
                values=rect,
                dtype=cp.int32,
            )

        if use_external_base_tensor:
            base_tensor_op = ChannelTensorOp(
                self,
                out_port_name="base",
                out_tensor_name="base",
                values=(0.0, 0.0, 50.0),
            )

        if use_external_scale_tensor:
            scale_tensor_op = ChannelTensorOp(
                self,
                out_port_name="scale",
                out_tensor_name="scale",
                values=(1.5, 1.0, 1.0),
            )

        visualizer = HolovizOp(
            self,
            name="holoviz",
            height=height_viz,
            width=width_viz,
            tensors=[dict(name="image", type="color", opacity=1.0, priority=0)],
        )

        self.add_flow(source, image_processing, {("output", "in_image")})
        if use_external_rect_tensor:
            self.add_flow(rect_tensor_op, image_processing, {("rect", "rect")})
        if use_external_base_tensor:
            self.add_flow(base_tensor_op, image_processing, {("base", "base")})
        if use_external_scale_tensor:
            self.add_flow(scale_tensor_op, image_processing, {("scale", "scale")})
        # self.add_flow()
        self.add_flow(image_processing, visualizer, {("out_image", "receivers")})


if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser(description="CV-CUDA demo application.")
    parser.add_argument(
        "-d",
        "--data",
        default="none",
        help=("Set the data path"),
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        default=0,
        help=("Number of frames to play (0 = run until user closes the window)"),
    )
    args = parser.parse_args()
    app = MyVideoProcessingApp(data=args.data, count=args.count)
    app.run()
