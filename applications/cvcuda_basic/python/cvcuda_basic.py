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

import cupy as cp
import numpy as np
import os
from argparse import ArgumentParser

import cvcuda
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import HolovizOp, VideoStreamReplayerOp
from holoscan.schedulers import EventBasedScheduler

from cropflipnormalizereformat import CropFlipNormalizeReformatOp
from cuosd import OnScreenDisplayOp
from cvtcolor import ColorConversionOp


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


#On-screen display elements
osd_elements = cvcuda.Elements(
    elements=[
        [
            cvcuda.BndBoxI(
                box=(10, 10, 25, 25),
                thickness=2,
                borderColor=(255, 255, 0),
                fillColor=(0, 128, 255, 128),  # blue semi-transparent
            ),
            cvcuda.BndBoxI(
                box=(200, 50, 20, 15),
                thickness=1,
                borderColor=(255, 255, 0),
                fillColor=(0, 0, 0, 0),  # fully transparent fill
            ),
            cvcuda.Label(
                utf8Text="label",
                fontSize=16,
                tlPos=(650, 50),
                fontColor=(255, 255, 0),
                bgColor=(0, 128, 255, 128),
            ),
            cvcuda.Segment(
                box=(80, 20, 50, 50),
                thickness=1,
                segArray=np.array(
                    [
                        [0, 0, 0, 0, 0.2, 0.2, 0, 0, 0, 0],
                        [0, 0, 0, 0.2, 0.3, 0.3, 0.2, 0, 0, 0],
                        [0, 0, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0, 0],
                        [0, 0.2, 0.3, 0.4, 0.5, 0.5, 0.4, 0.3, 0.2, 0],
                        [0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.4, 0.3, 0.2],
                        [0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.4, 0.3, 0.2],
                        [0, 0.2, 0.3, 0.4, 0.5, 0.5, 0.4, 0.3, 0.2, 0],
                        [0, 0, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0, 0],
                        [0, 0, 0, 0.2, 0.3, 0.3, 0.2, 0, 0, 0],
                        [0, 0, 0, 0, 0.2, 0.2, 0, 0, 0, 0],
                    ]
                ),
                segThreshold=0.2,
                borderColor=(255, 255, 255),
                segColor=(0, 128, 255, 128),
            ),
            cvcuda.Point(
                centerPos=(100, 100),
                radius=4,
                color=(255, 255, 0),
            ),
            cvcuda.Line(
                pos0=(80, 85),
                pos1=(150, 75),
                thickness=3,
                color=(255, 0, 0),
            ),
            cvcuda.PolyLine(
                points=np.array(
                    [
                        [120, 120],
                        [300, 100],
                        [250, 140],
                        [350, 180],
                        [400, 220],
                    ]
                ),
                thickness=0,
                isClosed=True,
                borderColor=(255, 255, 255),
                fillColor=(0, 255, 128, 96),
            ),
            cvcuda.RotatedBox(
                centerPos=(120, 140),
                width=12,
                height=12,
                yaw=0.3,
                thickness=1,
                borderColor=(255, 255, 0),
                bgColor=(0, 128, 255, 128),
            ),
            cvcuda.Circle(
                centerPos=(540, 30),
                radius=25,
                thickness=2,
                borderColor=(255, 255, 0),
                bgColor=(0, 128, 255, 128),
            ),
            cvcuda.Arrow(
                pos0=(550, 200),
                pos1=(450, 100),
                arrowSize=12,
                thickness=3,
                color=(0, 0, 255, 128),
            ),
            cvcuda.Clock(
                clockFormat=cvcuda.ClockFormat.YYMMDD_HHMMSS,
                time=0,
                fontSize=10,
                tlPos=(750, 210),
                fontColor=(255, 255, 0),
                bgColor=(0, 128, 255, 128),
            ),
        ],
    ],
)



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
            realtime=False,
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
            name="crop_flip_normalize_reformat",
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

        osd_op = OnScreenDisplayOp(self, name="osd", elements=osd_elements, in_layout="HWC")

        convert_to_grayscale = False
        if convert_to_grayscale:
            cvt_op = ColorConversionOp(
                self,
                name="rgb2gray",
                in_layout="HWC",
                code="RGB2GRAY",  # or code=cvcuda.ColorConversion.RGB2GRAY,
                out_format=cvcuda.Format.U8,
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
        self.add_flow(image_processing, osd_op, {("out_image", "in")})
        if convert_to_grayscale:
            self.add_flow(osd_op, cvt_op, {("out", "in")})
            self.add_flow(cvt_op, visualizer, {("out", "receivers")})
        else:
            self.add_flow(osd_op, visualizer, {("out", "receivers")})


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
    if False:
        app.scheduler(
            EventBasedScheduler(
                app,
                worker_thread_number=4,
                stop_on_deadlock=True,
                stop_on_deadlock_timeout=500,
                name="ebs",
            )
        )
    app.run()
