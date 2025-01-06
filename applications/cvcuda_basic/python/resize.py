"""
SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import cupy as cp

import cvcuda
import nvcv
from holoscan.core import Operator, OperatorSpec

__all__ = ["ResizeOp"]

interp_code_dict = {
    "nearest": cvcuda.Interp.NEAREST,
    "linear": cvcuda.Interp.LINEAR,
    "cubic": cvcuda.Interp.CUBIC,
    "area": cvcuda.Interp.AREA,
}


# Define custom Operators for use in the demo
class ResizeOp(Operator):
    """Image resize operator

     This operator has:
         inputs:  "in" (tensormap)
            Should have a tensormap with a single tensor of any name.
         outputs: "out" (tensormap)
            Emits a tensormap with a single tensor named 'image'.

    Each input frame is processed by CV-CUDA's resize operator.

    The data types supported for the input tensor are: uint8, uint16, int16, float32.
    The input tensor must have either 1, 3 or 4 channels.
    The layout of the input tensor must be one of HW, HWC or NHWC format.

    The output tensor will have the same data type and number of channels as the input tensor.

    Parameters
    ----------
    shape : tuple[int, int]
        Dimensions, width & height, of resized tensor.
    interp : {"nearest", "linear", "cubic", "area"} or cvcuda.Interp
        Interpolation type used for resizing. Note that "area" is only a suitable choice for downsampling.
    out_tensor_format : {"HW", "HWC", "NHWC"}, optional
        The format of the output tensor. Must match the input tensor aside from possibly adding
        or dropping singleton channel and batch dimensions.
    name : str, optional
        The name of the operator.

    Notes
    -----
    Since all images in an ImageBatchVarShape are resized to the same size, the resulting
    collection now fits in a single tensor.

    """  # noqa: E501

    def __init__(
        self,
        fragment,
        *args,
        shape,
        interp,
        out_tensor_format=None,
        name="resize",
        **kwargs,
    ):
        self.shape = shape
        self._prep_interp(interp)

        if out_tensor_format is not None:
            if out_tensor_format not in ["HW", "HWC", "NHWC"]:
                raise ValueError(f"unsupported out_tensor_format: {out_tensor_format}. "
                                 "Must be one of {'HW', 'HWC', 'NHWC'}")
        self.out_tensor_format = out_tensor_format

        # output CuPy array (to be set in compute method)
        self.cupy_out = None
        # output cvcuda.Tensor view of cupy_out (shares same memory pointer)
        self.cv_out = None

        # Need to call the base class constructor last
        super().__init__(fragment, *args, name=name, **kwargs)

    def _prep_interp(self, interp):
        """set cvcuda.Interp enum value."""
        # convert border name to cvcuda.Border type
        if isinstance(interp, cvcuda.Interp):
            self.interp = interp
        elif not isinstance(interp, str) or interp.lower() not in interp_code_dict:
            raise ValueError(f"Invalid interp value. Must be one of {tuple(interp_code_dict.keys())}")
        self.interp = interp_code_dict[interp]

    def _set_in_axis_indices(self, in_tensor_format):
        """Determine which array axes corresponds to each dimension (N, H, W, C).

        Any axes that are not present in the format string will be set to -1.
        """
        self.in_index_n = in_tensor_format.find("N")
        self.in_index_c = in_tensor_format.find("C")
        self.in_index_h = in_tensor_format.find("H")
        self.in_index_w = in_tensor_format.find("W")

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        tensormap = op_input.receive("in")
        if len(tensormap) != 1:
            raise ValueError("Input tensor map must have exactly one tensor.")
        input_tensor = tensormap.popitem()[1]

        if input_tensor.ndim == 2:
            self.in_tensor_format = "HW"
            input_tensor = cp.asarray(input_tensor)
            input_tensor = input_tensor.reshape(input_tensor.shape + (1,))
            self.in_tensor_layout = "HWC"
        elif input_tensor.ndim == 3:
            self.in_tensor_format = "HWC"
            self.in_tensor_layout = "HWC"
            self.original_in_tensor_format = "HWC"
        elif input_tensor.ndim == 4:
            self.in_tensor_format = "NHWC"
            self.in_tensor_layout = "NHWC"
            self.original_in_tensor_format = "NHWC"
        else:
            raise ValueError("Input tensor should have HW, HWC or NHWC dimensions")
        if self.out_tensor_format is None:
            self.out_tensor_format = self.in_tensor_format

        self._set_in_axis_indices(self.in_tensor_layout)

        # determine number of channels from input tensor
        num_channels = 1 if self.in_index_c == -1 else input_tensor.shape[self.in_index_c]
        # determine batch size from the input tensor
        num_batch = 1 if self.in_index_n == -1 else input_tensor.shape[self.in_index_n]

        cv_in_tensor = nvcv.as_tensor(input_tensor, layout=self.in_tensor_format)
        # cv_in = cvcuda.as_image(cv_in_tensor.cuda())
        # TODO: support multiple tensors and use the ImageBatchVarShape overload in that case

        # allocate the output tensor
        if self.cv_out is None:
            # determine output height and width from input tensor or specified rect argument
            height_width = self.shape

            # output tensor with same num_batch and num_channels as the input
            if self.in_tensor_layout == "HWC":
                out_shape = (height_width[0], height_width[1], num_channels)
            elif self.in_tensor_layout == "NHWC":
                out_shape = (num_batch, num_channels, height_width[0], height_width[1])

            # cast to CuPy array to get the CuPy dtype (input_tensor.dtype is the DLPack DLDataType)
            self.out_tensor_dtype = cp.asarray(input_tensor).dtype

            # cv_out will share cupy_out's memory so assign it to the class so it stays alive
            self.cupy_out = cp.empty(out_shape, dtype=self.out_tensor_dtype)
            self.cv_out = cvcuda.as_tensor(self.cupy_out, layout=self.in_tensor_layout)

        # use the already-allocated output buffer (self.cv_out)
        cvcuda.resize_into(
            dst=self.cv_out,
            src=cv_in_tensor,
            interp=self.interp,
            stream=None,  # TODO: use internal stream
        )

        # drop N and/or C dimensions to match the specified output format
        if self.out_tensor_format == "HW":
            if self.cupy_out.shape[-1] != 1:
                raise ValueError(
                    "Cannot drop channels dimension when it has size greater than 1."
                    )
            if self.out_tensor_layout == "HWC":
                out_cupy = self.cupy_out[..., 0]
            else:  # NHWC
                if self.cupy_out.shape[0] != 1:
                    raise ValueError(
                        "Cannot drop batch dimension when it has size greater than 1."
                    )
                out_cupy = self.cupy_out[0, ..., 0]
        elif self.out_tensor_format == "HWC" and "N" in self.in_tensor_layout:
            if self.cupy_out.shape[0] != 1:
                raise ValueError(
                    "Cannot drop batch dimension when it has size greater than 1."
                )
            out_cupy = self.cupy_out[0, ...]
        else:
            out_cupy = self.cupy_out

        output_tensormap = dict(image=out_cupy)
        op_output.emit(output_tensormap, "out")
