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
import os
from argparse import ArgumentParser

from holoscan.core import Application
from holoscan.operators import HolovizOp, VideoStreamReplayerOp

from resize import ResizeOp


# Now define a simple application using the operators defined above
class MyVideoProcessingApp(Application):
    """Example of an application that uses the operators defined above.

    This application has the following operators:

    - VideoStreamReplayerOp
    - ResizeOp
    - HolovizOp

    The VideoStreamReplayerOp reads a video file and sends the frames to the CropFlipNormalizeReformatOp.
    The ResizeOp resizes the frames and sends the processed frames to the HolovizOp.
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

        # upscale and round to size that is a multiple of 4
        shape = (int(height * 1.5), int(width * 1.5))
        shape = tuple(s // 4 * 4 for s in shape)
        image_processing = ResizeOp(
            self,
            shape=shape,
            interp="cubic",
            out_tensor_format="HWC",
            stream=None,
            name="resize",
        )

        visualizer = HolovizOp(
            self,
            name="holoviz",
            height=shape[0],
            width=shape[1],
            tensors=[dict(name="image", type="color", opacity=1.0, priority=0)],
        )

        self.add_flow(source, image_processing, {("output", "in")})
        self.add_flow(image_processing, visualizer, {("out", "receivers")})


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
