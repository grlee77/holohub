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

from holoscan.core import Operator, OperatorSpec

__all__ = ["ColorConversionOp"]


# get string names for the full set of enums available via cvcuda
_all_conversions = {k: v for k, v in cvcuda.ColorConversion.__dict__.items()
                    if isinstance(v, cvcuda.ColorConversion)}


# Groupings of codes of similar kinds with a tuple describing the dtypes supported
#   Determined by inspection of
#     - NVCVColorConversionCode enum from Types.h
#     - cvt_color.cu kernel implementations (for which codes are actually implemented)
rgb_codes = (
    "BGR2BGRA",   # 0  (add alpha channel)
    "RGB2RGBA",   # 0
    "BGRA2BGR",   # 1  (remove alpha channel)
    "RGBA2RGB",   # 1
    "BGR2RGBA",   # 2  (convert and add alpha)
    "RGB2BGRA",   # 2
    "RGBA2BGR",   # 3  (convert and remove alpha)
    "BGRA2RGB",   # 3
    "BGR2RGB",    # 4  (convert without alpha)
    "RGB2BGR",    # 4
    "BGRA2RGBA",  # 5  (convert with alpha)
    "RGBA2BGRA",  # 5
)
rgb_dtypes = (
    cp.int8, cp.int16, cp.int32, cp.uint8, cp.uint16, cp.float16, cp.float32, cp.float64
)

rgb_to_gray_codes = (
    "BGR2GRAY",     # 6
    "RGB2GRAY",     # 7
    # "BGRA2GRAY,   # 10
    # "RGBA2GRAY,   # 11
)
rgb_to_gray_dtypes = (cp.uint8, cp.uint16, cp.float32)

gray_to_rgb_codes = (
    "GRAY2BGR",     # 8 (gray to 3 channel)
    "GRAY2RGB",     # 8
    # "GRAY2RGBA",  # 9 (gray to 4 channel)
    # "GRAY2BGRA",  # 9
)
gray_to_rgb_dtypes = (
    cp.int8, cp.int16, cp.int32, cp.uint8, cp.uint16, cp.float16, cp.float32, cp.float64
)

hsv_codes = (
    "BGR2HSV",  # 40   (range 180)
    "RGB2HSV",  # 41
    "HSV2BGR",  # 54
    "HSV2RGB",  # 55
    "BGR2HSV_FULL",  # 66  (range 255)
    "RGB2HSV_FULL",  # 67
    "HSV2BGR_FULL",  # 70
    "HSV2RGB_FULL",  # 71
)
hsv_dtypes = (cp.uint8, cp.float32)

yuv_codes = (
    "BGR2YUV",  # 82
    "RGB2YUV",  # 83
    "YUV2BGR",  # 84
    "YUV2RGB",  # 85
)
yuv_dtypes = (cp.uint8, cp.uint16, cp.float32)

yuv420_codes = (
    # YUV 4:2:0 family to RGB
    "YUV2RGB_NV12",  # 90
    "YUV2BGR_NV12",  # 91
    "YUV2RGB_NV21",  # 92
    "YUV2BGR_NV21",  # 93
    "YUV420sp2RGB",  # YUV2RGB_NV21
    "YUV420sp2BGR",  # YUV2BGR_NV21

    "YUV2RGBA_NV12",  # 94
    "YUV2BGRA_NV12",  # 95
    "YUV2RGBA_NV21",  # 96
    "YUV2BGRA_NV21",  # 97
    "YUV420sp2RGBA",  # YUV2RGBA_NV21
    "YUV420sp2BGRA",  # YUV2BGRA_NV21

    "YUV2RGB_YV12",  # 98
    "YUV2BGR_YV12",  # 99
    "YUV2RGB_IYUV",  # 100
    "YUV2BGR_IYUV",  # 101
    "YUV2RGB_I420",  # YUV2RGB_IYUV
    "YUV2BGR_I420",  # YUV2BGR_IYUV
    "YUV420p2RGB",  # YUV2RGB_YV12
    "YUV420p2BGR",  # YUV2BGR_YV12

    "YUV2RGBA_YV12",  # 102,
    "YUV2BGRA_YV12",  # 103,
    "YUV2RGBA_IYUV",  # 104,
    "YUV2BGRA_IYUV",  # 105,
    "YUV2RGBA_I420",  # YUV2RGBA_IYUV
    "YUV2BGRA_I420",  # YUV2BGRA_IYUV
    "YUV420p2RGBA",  # YUV2RGBA_YV12
    "YUV420p2BGRA",  # YUV2BGRA_YV12

    "YUV2GRAY_420",  # 106,
    "YUV2GRAY_NV21",  # YUV2GRAY_420
    "YUV2GRAY_NV12",  # YUV2GRAY_420
    "YUV2GRAY_YV12",  # YUV2GRAY_420
    "YUV2GRAY_IYUV",  # YUV2GRAY_420
    "YUV2GRAY_I420",  # YUV2GRAY_420
    "YUV420sp2GRAY",  # YUV2GRAY_420
    "YUV420p2GRAY",  # YUV2GRAY_420

    # RGB to YUV 4:2:0 family (three plane YUV)
    "RGB2YUV_I420",  # 127,
    "BGR2YUV_I420",  # 128,
    "RGB2YUV_IYUV",  # RGB2YUV_I420,
    "BGR2YUV_IYUV",  # BGR2YUV_I420,

    "RGBA2YUV_I420",  # 129,
    "BGRA2YUV_I420",  # 130,
    "RGBA2YUV_IYUV",  # RGBA2YUV_I420,
    "BGRA2YUV_IYUV",  # BGRA2YUV_I420,
    "RGB2YUV_YV12",  # 131,
    "BGR2YUV_YV12",  # 132,
    "RGBA2YUV_YV12",  # 133,
    "BGRA2YUV_YV12",  # 134,

    # RGB to YUV 4:2:0 family (two plane YUV)
    "RGB2YUV_NV12",  # 140,
    "BGR2YUV_NV12",  # 141,
    "RGB2YUV_NV21",  # 142,
    "RGB2YUV420sp",  # RGB2YUV_NV21,
    "BGR2YUV_NV21",  # 143,
    "BGR2YUV420sp",  # BGR2YUV_NV21,

    "RGBA2YUV_NV12",  # 144,
    "BGRA2YUV_NV12",  # 145,
    "RGBA2YUV_NV21",  # 146,
    "RGBA2YUV420sp",  # RGBA2YUV_NV21,
    "BGRA2YUV_NV21",  # 147,
    "BGRA2YUV420sp",  # BGRA2YUV_NV21,
)
yuv420_dtypes = (cp.uint8, )


yuv422_codes = (
    # YUV 4:2:2 family to RGB
    "YUV2RGB_UYVY",  # 107,
    "YUV2BGR_UYVY",  # 108,
    # YUV2RGB_VYUY",  # 109,
    # YUV2BGR_VYUY",  # 110,
    "YUV2RGB_Y422",  # YUV2RGB_UYVY,
    "YUV2BGR_Y422",  # YUV2BGR_UYVY,
    "YUV2RGB_UYNV",  # YUV2RGB_UYVY,
    "YUV2BGR_UYNV",  # YUV2BGR_UYVY,

    "YUV2RGBA_UYVY",  # 111,
    "YUV2BGRA_UYVY",  # 112,
    # YUV2RGBA_VYUY",  # 113,
    # YUV2BGRA_VYUY",  # 114,
    "YUV2RGBA_Y422",  # YUV2RGBA_UYVY,
    "YUV2BGRA_Y422",  # YUV2BGRA_UYVY,
    "YUV2RGBA_UYNV",  # YUV2RGBA_UYVY,
    "YUV2BGRA_UYNV",  # YUV2BGRA_UYVY,

    "YUV2RGB_YUY2",  # 115,
    "YUV2BGR_YUY2",  # 116,
    "YUV2RGB_YVYU",  # 117,
    "YUV2BGR_YVYU",  # 118,
    "YUV2RGB_YUYV",  # YUV2RGB_YUY2,
    "YUV2BGR_YUYV",  # YUV2BGR_YUY2,
    "YUV2RGB_YUNV",  # YUV2RGB_YUY2,
    "YUV2BGR_YUNV",  # YUV2BGR_YUY2,

    "YUV2RGBA_YUY2",  # 119,
    "YUV2BGRA_YUY2",  # 120,
    "YUV2RGBA_YVYU",  # 121,
    "YUV2BGRA_YVYU",  # 122,
    "YUV2RGBA_YUYV",  # YUV2RGBA_YUY2,
    "YUV2BGRA_YUYV",  # YUV2BGRA_YUY2,
    "YUV2RGBA_YUNV",  # YUV2RGBA_YUY2,
    "YUV2BGRA_YUNV",  # YUV2BGRA_YUY2,

    "YUV2GRAY_UYVY",  # 123,
    "YUV2GRAY_YUY2",  # 124,
    #  CV_YUV2GRAY_VYUY   ",  # CV_YUV2GRAY_UYVY,
    "YUV2GRAY_Y422",  # YUV2GRAY_UYVY,
    "YUV2GRAY_UYNV",  # YUV2GRAY_UYVY,
    "YUV2GRAY_YVYU",  # YUV2GRAY_YUY2,
    "YUV2GRAY_YUYV",  # YUV2GRAY_YUY2,
    "YUV2GRAY_YUNV",  # YUV2GRAY_YUY2,
)
yuv422_dtypes = (cp.uint8, )

# Create a dictionary with the subset of the codes that have been implemented.
#   Note: Multiple string codes can share the same enum value so better to
#         use strings as the keys than enum values.
implemented_conversions = {
    k: v for k, v in _all_conversions.items()
    if (k in rgb_codes or k in rgb_to_gray_codes or k in gray_to_rgb_codes or
        k in hsv_codes or k in yuv_codes or k in yuv420_codes or k in yuv422_codes)
}

# Update values in the implemented_conversions to be a 2-tuple with the
# CVCUDA enum as the first value and supported dtypes as the second
for k in rgb_codes:
    implemented_conversions[k] = (implemented_conversions[k], rgb_dtypes)
for k in rgb_to_gray_codes:
    implemented_conversions[k] = (implemented_conversions[k], rgb_to_gray_dtypes)
for k in gray_to_rgb_codes:
    implemented_conversions[k] = (implemented_conversions[k], gray_to_rgb_dtypes)
for k in hsv_codes:
    implemented_conversions[k] = (implemented_conversions[k], hsv_dtypes)
for k in yuv_codes:
    implemented_conversions[k] = (implemented_conversions[k], yuv_dtypes)
for k in yuv420_codes:
    implemented_conversions[k] = (implemented_conversions[k], yuv420_dtypes)
for k in yuv422_codes:
    implemented_conversions[k] = (implemented_conversions[k], yuv422_dtypes)


def print_implemented_conversions_tables():
    # determine maximum code name length
    max_code_name_length = max(len(k) for k in implemented_conversions)
    # determine maximum length of concatenated dtype names
    max_dtype_width = max(
        len(', '.join(cp.dtype(d).name for d in dtypes))
        for dtypes in (rgb_dtypes, rgb_to_gray_dtypes, gray_to_rgb_dtypes,
                       hsv_dtypes, yuv_dtypes, yuv420_dtypes, yuv422_dtypes)
    )
    # add two to leave at least 1 blank space on either side
    code_width = max_code_name_length + 2
    dtype_width = max_dtype_width + 2

    # Define common header with fixed width fields for all tables
    code_header = "Code"
    dtype_header = "Supported dtypes"
    table_header = f"|{code_header:^{code_width}}|{dtype_header:^{dtype_width}}|"
    table_header += "\n|" + "-" * code_width + "|" + "-" * dtype_width + "|"

    # Print tables with codes grouped by conversion types
    print("### Table of RGB <-> BGR conversions\n")
    print(table_header)
    for k in rgb_codes:
        print(f"| {k:<{code_width - 1}}| "
              f"{', '.join(cp.dtype(d).name for d in rgb_dtypes):<{dtype_width - 1}}|")

    print("\n### Table of RGB <-> Grayscale conversions\n")
    print(table_header)
    for k in rgb_to_gray_codes:
        print(f"| {k:<{code_width - 1}}| "
              f"{', '.join(cp.dtype(d).name for d in rgb_to_gray_dtypes):<{dtype_width - 1}}|")
    for k in gray_to_rgb_codes:
        print(f"| {k:<{code_width - 1}}| "
              f"{', '.join(cp.dtype(d).name for d in gray_to_rgb_dtypes):<{dtype_width - 1}}|")

    print("\n### Table of RGB <-> HSV conversions\n")
    print(table_header)
    for k in hsv_codes:
        print(f"| {k:<{code_width - 1}}| "
              f"{', '.join(cp.dtype(d).name for d in hsv_dtypes):<{dtype_width - 1}}|")

    print("\n### Table of RGB <-> YUV conversions\n")
    print(table_header)
    for k in yuv_codes:
        print(f"| {k:<{code_width - 1}}| "
              f"{', '.join(cp.dtype(d).name for d in yuv_dtypes):<{dtype_width - 1}}|")

    print("\n### Table of RGB <-> YUV 4:2:0 conversions\n")
    print(table_header)
    for k in yuv420_codes:
        print(f"| {k:<{code_width - 1}}| "
              f"{', '.join(cp.dtype(d).name for d in yuv420_dtypes):<{dtype_width - 1}}|")

    print("\n### Table of RGB <-> YUV 4:2:2 conversions\n")
    print(table_header)
    for k in yuv422_codes:
        print(f"| {k:<{code_width - 1}}| "
              f"{', '.join(cp.dtype(d).name for d in yuv422_dtypes):<{dtype_width - 1}}|")


class ColorConversionOp(Operator):
    def __init__(
        self,
        fragment,
        *args,
        in_layout="HWC",
        code="RGB2GRAY",  # src/cvcuda/include/cvcuda/Types.h
        name="color_convert",
        **kwargs,
    ):
        if in_layout not in ("HWC", "NHWC"):
            raise ValueError(f"in_layout ({in_layout}) must be either 'HWC' or 'NHWC'")
        if isinstance(code, cvcuda.ColorConversion):
            # trick to get string representation of the enum
            code = f"{code}".split('.')[1]
        if isinstance(code, str):
            if code not in implemented_conversions:
                implemented_table = print_implemented_conversions_tables()
                if code in _all_conversions:
                    raise ValueError(
                        f"code ({code}) exists but is not yet implemented. See table below of "
                        f"supported cases: {implemented_table}"
                    )
                else:
                    raise ValueError(
                        f"unrecognized code ({code}). See table below of supported cases: "
                        f"{implemented_table}"
                    )
            self.code_name = code
        else:
            raise ValueError(
                "expected code to be a str or a cvcuda.ColorConversion enum value"
            )
        self.in_layout = in_layout
        self.code, self.supported_dtypes = implemented_conversions[self.code_name]
        self.out = None
        super().__init__(fragment, *args, name=name, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        tensormap = op_input.receive("in")
        if len(tensormap) != 1:
            raise ValueError("Input tensor map must have exactly one tensor.")
        image_in = cp.asarray(tensormap.popitem()[1])

        # on the first frame only, validate the dtype of the input frame
        if self.out is None:
            if image_in.dtype not in self.supported_dtypes:
                raise RuntimeError(
                    f"image_in.dtype ({image_in.dtype.name}) for code '{self.code_name}' should have "
                    "one of the following dtypes: ("
                    f"{', '.join(cp.dtype(d).name for d in self.supported_dtypes)}).")

        if image_in.ndim != len(self.in_layout):
            raise ValueError(
                f"input tensor is {image_in.ndim}d but layout would correspond "
                f"to a {len(self.in_layout)}d tensor"
            )
        cv_image_in = cvcuda.as_tensor(image_in, layout=self.in_layout)

        if self.out is None:
            # TODO: use internal stream
            self.out = cvcuda.cvtcolor(src=cv_image_in, code=self.code, stream=None)
            self.out_cupy = cp.asarray(self.out.cuda())
        else:
            cvcuda.cvtcolor_into(dst=self.out, src=cv_image_in, code=self.code, stream=self.stream)
        op_output.emit(dict(image=self.out_cupy), "out")
