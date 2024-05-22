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


# Reference
rgb_codes = (
    cvcuda.ColorConversion.BGR2BGRA,   # 0  (add alpha channel)
    cvcuda.ColorConversion.RGB2RGBA,   # 0
    cvcuda.ColorConversion.BGRA2BGR,   # 1  (remove alpha channel)
    cvcuda.ColorConversion.RGBA2RGB,   # 1
    cvcuda.ColorConversion.BGR2RGBA,   # 2  (convert and add alpha)
    cvcuda.ColorConversion.RGB2BGRA,   # 2
    cvcuda.ColorConversion.RGBA2BGR,   # 3  (convert and remove alpha)
    cvcuda.ColorConversion.BGRA2RGB,   # 3
    cvcuda.ColorConversion.BGR2RGB,    # 4  (convert without alpha)
    cvcuda.ColorConversion.RGB2BGR,    # 4
    cvcuda.ColorConversion.BGRA2RGBA,  # 5  (convert with alpha)
    cvcuda.ColorConversion.RGBA2BGRA,  # 5
)
rgb_dtypes = (
    cp.int8, cp.int16, cp.int32, cp.uint8, cp.uint16, cp.float16, cp.float32, cp.float64
)

rgb_to_gray_codes = (
    cvcuda.ColorConversion.BGR2GRAY,     # 6
    cvcuda.ColorConversion.RGB2GRAY,     # 7
    # cvcuda.ColorConversion.BGRA2GRAY,  # 10
    # cvcuda.ColorConversion.RGBA2GRAY,  # 11
)
rgb_to_gray_dtypes = (cp.uint8, cp.uint16, cp.float32)

gray_to_rgb_codes = (
    cvcuda.ColorConversion.GRAY2BGR,     # 8 (gray to 3 channel)
    cvcuda.ColorConversion.GRAY2RGB,     # 8
    # cvcuda.ColorConversion.GRAY2RGBA,  # 9 (gray to 4 channel)
    # cvcuda.ColorConversion.GRAY2BGRA,  # 9
)
gray_to_rgb_dtypes = (
    cp.int8, cp.int16, cp.int32, cp.uint8, cp.uint16, cp.float16, cp.float32, cp.float64
)

hsv_codes = (
    cvcuda.ColorConversion.BGR2HSV,  # 40   (range 180)
    cvcuda.ColorConversion.RGB2HSV,  # 41
    cvcuda.ColorConversion.HSV2BGR,  # 54
    cvcuda.ColorConversion.HSV2RGB,  # 55
    cvcuda.ColorConversion.BGR2HSV_FULL,  # 66  (range 255)
    cvcuda.ColorConversion.RGB2HSV_FULL,  # 67
    cvcuda.ColorConversion.HSV2BGR_FULL,  # 70
    cvcuda.ColorConversion.HSV2RGB_FULL,  # 71
)
hsv_dtypes = (cp.uint8, cp.float32)

yuv_codes = (
    cvcuda.ColorConversion.BGR2YUV,  # 82
    cvcuda.ColorConversion.RGB2YUV,  # 83
    cvcuda.ColorConversion.YUV2BGR,  # 84
    cvcuda.ColorConversion.YUV2RGB,  # 85
)
yuv_dtypes = (cp.uint8, cp.uint16, cp.float32)

yuv420_codes = (
    # YUV 4:2:0 family to RGB
    cvcuda.ColorConversion.YUV2RGB_NV12,  # 90
    cvcuda.ColorConversion.YUV2BGR_NV12,  # 91
    cvcuda.ColorConversion.YUV2RGB_NV21,  # 92
    cvcuda.ColorConversion.YUV2BGR_NV21,  # 93
    cvcuda.ColorConversion.YUV420sp2RGB,  # cvcuda.ColorConversion.YUV2RGB_NV21
    cvcuda.ColorConversion.YUV420sp2BGR,  # cvcuda.ColorConversion.YUV2BGR_NV21

    cvcuda.ColorConversion.YUV2RGBA_NV12,  # 94
    cvcuda.ColorConversion.YUV2BGRA_NV12,  # 95
    cvcuda.ColorConversion.YUV2RGBA_NV21,  # 96
    cvcuda.ColorConversion.YUV2BGRA_NV21,  # 97
    cvcuda.ColorConversion.YUV420sp2RGBA,  # cvcuda.ColorConversion.YUV2RGBA_NV21
    cvcuda.ColorConversion.YUV420sp2BGRA,  # cvcuda.ColorConversion.YUV2BGRA_NV21

    cvcuda.ColorConversion.YUV2RGB_YV12,  # 98
    cvcuda.ColorConversion.YUV2BGR_YV12,  # 99
    cvcuda.ColorConversion.YUV2RGB_IYUV,  # 100
    cvcuda.ColorConversion.YUV2BGR_IYUV,  # 101
    cvcuda.ColorConversion.YUV2RGB_I420,  # cvcuda.ColorConversion.YUV2RGB_IYUV
    cvcuda.ColorConversion.YUV2BGR_I420,  # cvcuda.ColorConversion.YUV2BGR_IYUV
    cvcuda.ColorConversion.YUV420p2RGB ,  # cvcuda.ColorConversion.YUV2RGB_YV12
    cvcuda.ColorConversion.YUV420p2BGR ,  # cvcuda.ColorConversion.YUV2BGR_YV12

    cvcuda.ColorConversion.YUV2RGBA_YV12,  # 102,
    cvcuda.ColorConversion.YUV2BGRA_YV12,  # 103,
    cvcuda.ColorConversion.YUV2RGBA_IYUV,  # 104,
    cvcuda.ColorConversion.YUV2BGRA_IYUV,  # 105,
    cvcuda.ColorConversion.YUV2RGBA_I420,  # cvcuda.ColorConversion.YUV2RGBA_IYUV
    cvcuda.ColorConversion.YUV2BGRA_I420,  # cvcuda.ColorConversion.YUV2BGRA_IYUV
    cvcuda.ColorConversion.YUV420p2RGBA ,  # cvcuda.ColorConversion.YUV2RGBA_YV12
    cvcuda.ColorConversion.YUV420p2BGRA ,  # cvcuda.ColorConversion.YUV2BGRA_YV12

    cvcuda.ColorConversion.YUV2GRAY_420 ,  # 106,
    cvcuda.ColorConversion.YUV2GRAY_NV21,  # cvcuda.ColorConversion.YUV2GRAY_420
    cvcuda.ColorConversion.YUV2GRAY_NV12,  # cvcuda.ColorConversion.YUV2GRAY_420
    cvcuda.ColorConversion.YUV2GRAY_YV12,  # cvcuda.ColorConversion.YUV2GRAY_420
    cvcuda.ColorConversion.YUV2GRAY_IYUV,  # cvcuda.ColorConversion.YUV2GRAY_420
    cvcuda.ColorConversion.YUV2GRAY_I420,  # cvcuda.ColorConversion.YUV2GRAY_420
    cvcuda.ColorConversion.YUV420sp2GRAY,  # cvcuda.ColorConversion.YUV2GRAY_420
    cvcuda.ColorConversion.YUV420p2GRAY ,  # cvcuda.ColorConversion.YUV2GRAY_420

    # RGB to YUV 4:2:0 family (three plane YUV)
    cvcuda.ColorConversion.RGB2YUV_I420,  # 127,
    cvcuda.ColorConversion.BGR2YUV_I420,  # 128,
    cvcuda.ColorConversion.RGB2YUV_IYUV,  # cvcuda.ColorConversion.RGB2YUV_I420,
    cvcuda.ColorConversion.BGR2YUV_IYUV,  # cvcuda.ColorConversion.BGR2YUV_I420,

    cvcuda.ColorConversion.RGBA2YUV_I420,  # 129,
    cvcuda.ColorConversion.BGRA2YUV_I420,  # 130,
    cvcuda.ColorConversion.RGBA2YUV_IYUV,  # cvcuda.ColorConversion.RGBA2YUV_I420,
    cvcuda.ColorConversion.BGRA2YUV_IYUV,  # cvcuda.ColorConversion.BGRA2YUV_I420,
    cvcuda.ColorConversion.RGB2YUV_YV12 ,  # 131,
    cvcuda.ColorConversion.BGR2YUV_YV12 ,  # 132,
    cvcuda.ColorConversion.RGBA2YUV_YV12,  # 133,
    cvcuda.ColorConversion.BGRA2YUV_YV12,  # 134,

    # RGB to YUV 4:2:0 family (two plane YUV)
    cvcuda.ColorConversion.RGB2YUV_NV12,  # 140,
    cvcuda.ColorConversion.BGR2YUV_NV12,  # 141,
    cvcuda.ColorConversion.RGB2YUV_NV21,  # 142,
    cvcuda.ColorConversion.RGB2YUV420sp,  # cvcuda.ColorConversion.RGB2YUV_NV21,
    cvcuda.ColorConversion.BGR2YUV_NV21,  # 143,
    cvcuda.ColorConversion.BGR2YUV420sp,  # cvcuda.ColorConversion.BGR2YUV_NV21,

    cvcuda.ColorConversion.RGBA2YUV_NV12,  # 144,
    cvcuda.ColorConversion.BGRA2YUV_NV12,  # 145,
    cvcuda.ColorConversion.RGBA2YUV_NV21,  # 146,
    cvcuda.ColorConversion.RGBA2YUV420sp,  # cvcuda.ColorConversion.RGBA2YUV_NV21,
    cvcuda.ColorConversion.BGRA2YUV_NV21,  # 147,
    cvcuda.ColorConversion.BGRA2YUV420sp,  # cvcuda.ColorConversion.BGRA2YUV_NV21,
)
yuv420_dtypes = (cp.uint8, )


yuv422_codes = (
    # YUV 4:2:2 family to RGB
    cvcuda.ColorConversion.YUV2RGB_UYVY,  # 107,
    cvcuda.ColorConversion.YUV2BGR_UYVY,  # 108,
    # cvcuda.ColorConversion.YUV2RGB_VYUY,  # 109,
    # cvcuda.ColorConversion.YUV2BGR_VYUY,  # 110,
    cvcuda.ColorConversion.YUV2RGB_Y422,  # cvcuda.ColorConversion.YUV2RGB_UYVY,
    cvcuda.ColorConversion.YUV2BGR_Y422,  # cvcuda.ColorConversion.YUV2BGR_UYVY,
    cvcuda.ColorConversion.YUV2RGB_UYNV,  # cvcuda.ColorConversion.YUV2RGB_UYVY,
    cvcuda.ColorConversion.YUV2BGR_UYNV,  # cvcuda.ColorConversion.YUV2BGR_UYVY,

    cvcuda.ColorConversion.YUV2RGBA_UYVY,  # 111,
    cvcuda.ColorConversion.YUV2BGRA_UYVY,  # 112,
    # cvcuda.ColorConversion.YUV2RGBA_VYUY,  # 113,
    # cvcuda.ColorConversion.YUV2BGRA_VYUY,  # 114,
    cvcuda.ColorConversion.YUV2RGBA_Y422,  # cvcuda.ColorConversion.YUV2RGBA_UYVY,
    cvcuda.ColorConversion.YUV2BGRA_Y422,  # cvcuda.ColorConversion.YUV2BGRA_UYVY,
    cvcuda.ColorConversion.YUV2RGBA_UYNV,  # cvcuda.ColorConversion.YUV2RGBA_UYVY,
    cvcuda.ColorConversion.YUV2BGRA_UYNV,  # cvcuda.ColorConversion.YUV2BGRA_UYVY,

    cvcuda.ColorConversion.YUV2RGB_YUY2,  # 115,
    cvcuda.ColorConversion.YUV2BGR_YUY2,  # 116,
    cvcuda.ColorConversion.YUV2RGB_YVYU,  # 117,
    cvcuda.ColorConversion.YUV2BGR_YVYU,  # 118,
    cvcuda.ColorConversion.YUV2RGB_YUYV,  # cvcuda.ColorConversion.YUV2RGB_YUY2,
    cvcuda.ColorConversion.YUV2BGR_YUYV,  # cvcuda.ColorConversion.YUV2BGR_YUY2,
    cvcuda.ColorConversion.YUV2RGB_YUNV,  # cvcuda.ColorConversion.YUV2RGB_YUY2,
    cvcuda.ColorConversion.YUV2BGR_YUNV,  # cvcuda.ColorConversion.YUV2BGR_YUY2,

    cvcuda.ColorConversion.YUV2RGBA_YUY2,  # 119,
    cvcuda.ColorConversion.YUV2BGRA_YUY2,  # 120,
    cvcuda.ColorConversion.YUV2RGBA_YVYU,  # 121,
    cvcuda.ColorConversion.YUV2BGRA_YVYU,  # 122,
    cvcuda.ColorConversion.YUV2RGBA_YUYV,  # cvcuda.ColorConversion.YUV2RGBA_YUY2,
    cvcuda.ColorConversion.YUV2BGRA_YUYV,  # cvcuda.ColorConversion.YUV2BGRA_YUY2,
    cvcuda.ColorConversion.YUV2RGBA_YUNV,  # cvcuda.ColorConversion.YUV2RGBA_YUY2,
    cvcuda.ColorConversion.YUV2BGRA_YUNV,  # cvcuda.ColorConversion.YUV2BGRA_YUY2,

    cvcuda.ColorConversion.YUV2GRAY_UYVY,  # 123,
    cvcuda.ColorConversion.YUV2GRAY_YUY2,  # 124,
    #  CV_YUV2GRAY_VYUY   ,  # CV_YUV2GRAY_UYVY,
    cvcuda.ColorConversion.YUV2GRAY_Y422,  # cvcuda.ColorConversion.YUV2GRAY_UYVY,
    cvcuda.ColorConversion.YUV2GRAY_UYNV,  # cvcuda.ColorConversion.YUV2GRAY_UYVY,
    cvcuda.ColorConversion.YUV2GRAY_YVYU,  # cvcuda.ColorConversion.YUV2GRAY_YUY2,
    cvcuda.ColorConversion.YUV2GRAY_YUYV,  # cvcuda.ColorConversion.YUV2GRAY_YUY2,
    cvcuda.ColorConversion.YUV2GRAY_YUNV,  # cvcuda.ColorConversion.YUV2GRAY_YUY2,
)
yuv422_dtypes = (cp.uint8, )


class ColorConversionOp(Operator):
    def __init__(
        self,
        fragment,
        *args,
        in_layout="HWC",
        code=cvcuda.ColorConversion.RGB2GRAY,  # src/cvcuda/include/cvcuda/Types.h
        stream=None,
        **kwargs,
    ):
        # checked RGB, BGR 2 Gray conversions and they work for
        #    u8, u16 or f32 dtype where input and output dtype must match
        #    layout of input must be either HWC or NHWC
        if in_layout not in ("HWC", "NHWC"):
            raise ValueError(f"in_layout ({in_layout}) must be either 'HWC' or 'NHWC'")
        self.in_layout = in_layout
        self.code = code
        self.stream = stream
        self.out = None
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        tensormap = op_input.receive("in")

        image_in = cp.asarray(tensormap["image"])
        if image_in.ndim != len(self.in_layout):
            raise ValueError(
                f"input tensor is {image_in.ndim}d but layout would correspond "
                f"to a {len(self.in_layout)}d tensor"
            )
        cv_image_in = cvcuda.as_tensor(image_in, layout=self.in_layout)

        if self.out is None:
            self.out = cvcuda.cvtcolor(src=cv_image_in, code=self.code, stream=self.stream)
            self.out_cupy = cp.asarray(self.out.cuda())
        else:
            cvcuda.cvtcolor_into(dst=self.out, src=cv_image_in, code=self.code, stream=self.stream)
        # print(f"{self.out_cupy.shape=}")
        op_output.emit(dict(image=self.out_cupy), "out")
