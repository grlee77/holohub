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

__all__ = ["HQResizeOp"]

interp_code_dict = {
    "nearest": cvcuda.Interp.NEAREST,
    "linear": cvcuda.Interp.LINEAR,
    "cubic": cvcuda.Interp.CUBIC,
    "lanczos": cvcuda.Interp.LANCZOS,
    "gaussian": cvcuda.Interp.GAUSSIAN,
}


# Define custom Operators for use in the demo
class HQResizeOp(Operator):
    """Image or Volume resize operator (high quality version).

     This operator has:
         inputs:  "in" (tensormap)
            Should have a tensormap with a single tensor of any name.
         outputs: "out" (tensormap)
            Emits a tensormap with a single tensor named 'image'.

    Each input frame is processed by CV-CUDA's hq_resize operator.

    The data types supported for the input tensor are: uint8, uint16, int16, float32.
    The input tensor must have either 1, 3 or 4 channels.
    The layout of the input tensor must be one of HW, HWC or NHWC format.

    The output tensor must have either the same data type as the input or have float32
    data type.

    Parameters
    ----------
    shape : 2-tuple or 3-tuple of int
        Dimensions (D)HW, of a 2D image or 3D volume.
    roi : 4-tuple or 6-tuple of int
        Region of interest to resize. This is specified as a tuple of low and high values into the
        input image. The region corresponding to the ROI will be resized to the specified `shape`.
        If None, the full input image is used. The format is as follows::

            - For 2D images: (h_low, w_low, h_high, w_high)
            - For 3D volumes: (d_low, h_low, w_low, d_high, h_high, w_high)

        If, for some axis, the low value is higher than the high value, the tensor will be flipped
        along that axis.
    antialias : bool, optional
        If True, apply an antialiasing filter before downscaling.
    interpolation : {"nearest", "linear", "cubic", "lanczos", "gaussian"}, optional
        Interpolation type used for resizing. If specified this is used for both downscaling
        and upscaling and `min_interpolation` and `mag_interpolation` are ignored.
    min_interpolation : {"nearest", "linear", "cubic", "lanczos", "gaussian"}, optional
        Interpolation type used for downscaling.
    max_interpolation : {"nearest", "linear", "cubic", "lanczos", "gaussian"}, optional
        Interpolation type used for upscaling.
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
        in_tensor_format,
        antialias=True,
        interpolation=None,
        min_interpolation=None,
        mag_interpolation=None,
        roi=None,
        out_tensor_format=None,
        name="hqresize",
        **kwargs,
    ):
        # name of the input port
        self.in_tensor_format = in_tensor_format
        self.ndim_spatial = 3 if "D" in self.in_tensor_format else 2

        # shape includes only spatial dimensions (not batch or channels)
        self.shape = shape
        self.antialias = antialias
        self._prep_interp(interpolation, min_interpolation, mag_interpolation)
        self._prep_roi(roi, self.ndim_spatial)

        if out_tensor_format is not None:
            if out_tensor_format not in ["HW", "HWC", "NHWC", "DHW", "DHWC", "NDHWC"]:
                raise ValueError(f"unsupported out_tensor_format: {out_tensor_format}. "
                                 "Must be one of {'HW', 'HWC', 'NHWC', 'DHW', 'DHWC', 'NDHWC'}")
        self.out_tensor_format = out_tensor_format

        # output CuPy array (to be set in compute method)
        self.cupy_out = None
        # output cvcuda.Tensor view of cupy_out (shares same memory pointer)
        self.cv_out = None

        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def _check_interp(self, interp):
        # convert border name to cvcuda.Border type
        if isinstance(interp, cvcuda.Interp):
            return interp
        elif not isinstance(interp, str) or interp.lower() not in interp_code_dict:
            raise ValueError(f"Invalid interp value. Must be one of {tuple(interp_code_dict.keys())}")
        return interp_code_dict[interp]

    def _prep_roi(self, roi, ndim_spatial):
        if roi is not None:
            if ndim_spatial == 2:
                if len(roi) != 4:
                    raise ValueError(
                        "For 2D images, the ROI must be a 4-tuple: (h_low, w_low, h_high, w_high)."
                    )
            else:
                if len(roi) != 6:
                    raise ValueError(
                        "For 3D images, the ROI must be a 6-tuple: "
                        "(d_low, h_low, w_low, d_low, h_high, w_high)."
                    )
        self.roi = roi

    def _prep_interp(self, interpolation, min_interpolation, mag_interpolation):
        """set cvcuda.Interp enum value."""
        if interpolation is not None:
            if min_interpolation is not None or mag_interpolation is not None:
                raise ValueError(
                    "Cannot specify both interpolation and min_interpolation/mag_interpolation"
                )
            interpolation = self._check_interp(interpolation)
            self.min_interpolation = interpolation
            self.mag_interpolation = interpolation
        else:
            if min_interpolation is None or mag_interpolation is None:
                raise ValueError(
                    "Must specify either interpolation or both min_interpolation and mag_interpolation"
                )
            self.min_interpolation = self._check_interp(min_interpolation)
            self.mag_interpolation = self._check_interp(mag_interpolation)

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

    def start(self):
        if self.out_tensor_format is None:
            self.out_tensor_format = self.in_tensor_format

        self._set_in_axis_indices(self.in_tensor_format)

    def compute(self, op_input, op_output, context):
        tensormap = op_input.receive("in")
        if len(tensormap) != 1:
            raise ValueError("Input tensor map must have exactly one tensor.")
        input_tensor = tensormap.popitem()[1]

        ndim_expected = len(self.in_tensor_format)
        if input_tensor.ndim != len(self.in_tensor_format):
            raise RuntimeError(
                f"Tensor has {input_tensor.ndim} dimensions, but expected a tensor with "
                f"{ndim_expected} dimensions corresponding to {self.in_tensor_format} format.",
            )

        # determine number of channels from input tensor
        num_channels = 1 if self.in_index_c == -1 else input_tensor.shape[self.in_index_c]
        # determine batch size from the input tensor
        num_batch = 1 if self.in_index_n == -1 else input_tensor.shape[self.in_index_n]

        cv_in_tensor = nvcv.as_tensor(input_tensor, layout=self.in_tensor_format)
        # cv_in = cvcuda.as_image(cv_in_tensor.cuda())
        # TODO: support ImageBatchVarShape and TensorBatch variants as well

        # allocate the output tensor
        if self.cv_out is None:
            # determine output height and width from input tensor or specified rect argument
            if self.ndim_spatial == 2:
                height, width = self.shape
            else:
                depth, height, width = self.shape
            # output tensor with same num_batch and num_channels as the input
            if self.in_tensor_format in ["HW", "DHW"]:
                out_shape = self.shape
            elif self.in_tensor_format in ["HWC", "DHWC"]:
                out_shape = self.shape + (num_channels,)
            elif self.in_tensor_format in ["NHWC", "NDHWC"]:
                out_shape = (num_batch,) + self.shape + (num_channels,)

            # cast to CuPy array to get the CuPy dtype (input_tensor.dtype is the DLPack DLDataType)
            self.out_tensor_dtype = cp.asarray(input_tensor).dtype

            # cv_out will share cupy_out's memory so assign it to the class so it stays alive
            self.cupy_out = cp.empty(out_shape, dtype=self.out_tensor_dtype)
            self.cv_out = cvcuda.as_tensor(self.cupy_out, layout=self.in_tensor_format)

        # use the already-allocated output buffer (self.cv_out)
        cvcuda.hq_resize_into(
            dst=self.cv_out,
            src=cv_in_tensor,
            antialias=self.antialias,
            roi=self.roi,
            interpolation=None,
            min_interpolation=self.min_interpolation,
            mag_interpolation=self.mag_interpolation,
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
        elif self.out_tensor_format == "HWC" and "N" in self.in_tensor_format:
            if self.cupy_out.shape[0] != 1:
                raise ValueError(
                    "Cannot drop batch dimension when it has size greater than 1."
                )
            out_cupy = self.cupy_out[0, ...]
        else:
            out_cupy = self.cupy_out

        output_tensormap = dict(image=out_cupy)
        op_output.emit(output_tensormap, "out")
