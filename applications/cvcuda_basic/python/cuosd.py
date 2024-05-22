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
import numpy as np

import cvcuda
import nvcv
from holoscan.core import Operator, OperatorSpec


class cuOSDOp(Operator):
    def __init__(
        self,
        fragment,
        *args,
        elements,
        in_layout="HWC",
        **kwargs,
    ):
        self.elements = elements
        self.in_layout = in_layout
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
            self.out = cvcuda.osd(src=cv_image_in, elements=self.elements)
            self.out_cupy = cp.asarray(self.out.cuda())
        else:
            cvcuda.osd_into(dst=self.out, src=cv_image_in, elements=self.elements)
        op_output.emit(dict(image=self.out_cupy), "out")
