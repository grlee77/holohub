/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_runtime.h>
#include <getopt.h>

#include <cstdint>
#include <memory>
#include <string>

#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>
#include "cvcuda_to_holoscan.hpp"
#include "holoscan/holoscan.hpp"
#include "holoscan_to_cvcuda.hpp"

#include <cvcuda/OpHQResize.hpp>        // cvcuda::HQResize
#include <cvcuda/Workspace.hpp>         // cvcuda::UniqueWorkspace
#include <nvcv/alloc/Requirements.hpp>  // nvcv::CalcTotalSizeBytes
#include <nvcv/DataType.hpp>            // nvcv::DataType
#include <nvcv/Tensor.hpp>              // nvcv::Tensor
#include <nvcv/TensorData.hpp>          // nvcv::TensorDataStridedCuda
#include <nvcv/TensorShape.hpp>         // nvcv::TensorShape

namespace holoscan::ops {

namespace {

#define CUDA_TRY(stmt)                                                                  \
  {                                                                                     \
    cudaError_t cuda_status = stmt;                                                     \
    if (cudaSuccess != cuda_status) {                                                   \
      HOLOSCAN_LOG_ERROR("Runtime call {} in line {} of file {} failed with '{}' ({})", \
                         #stmt,                                                         \
                         __LINE__,                                                      \
                         __FILE__,                                                      \
                         cudaGetErrorString(cuda_status),                               \
                         static_cast<int>(cuda_status));                                \
      throw std::runtime_error("Error in CUDA library function call");                  \
    }                                                                                   \
  }

// helper copied from CV-CUDA's  TestOpHQResize
template <typename... Extents>
nvcv::Tensor CreateTensorHelper(nvcv::DataType dtype, const char* layoutStr, int numSamples,
                                Extents... extents) {
  nvcv::TensorLayout layout{layoutStr};
  if (numSamples == 1) {
    nvcv::TensorShape shape{{extents...}, layout.last(sizeof...(extents))};
    return nvcv::Tensor{shape, dtype};
  } else {
    nvcv::TensorShape shape{{numSamples, extents...}, layout};
    return nvcv::Tensor{shape, dtype};
  }
}

}  // namespace

// Apply basic CV-CUDA processing to the input frame
class HQResizeOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(HQResizeOp);
  HQResizeOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<nvcv::Tensor>("input_tensor");
    spec.output<nvcv::Tensor>("output_tensor");
    spec.param(shape_, "shape", "shape", "The shape after resizing");
    spec.param(
        in_tensor_format_,
        "in_tensor_format",
        "input tensor format",
        "input tensor format (must be one of {'HW', 'HWC', 'NHWC', 'DHW', 'DHWC', 'NDHWC'})");
    spec.param(
        out_tensor_format_,
        "out_tensor_format",
        "output tensor format",
        "output tensor format (must be one of {'HW', 'HWC', 'NHWC', 'DHW', 'DHWC', 'NDHWC'})",
        ParameterFlag::kOptional);
    spec.param(roi_, "roi", "ROI", "region-of-interest to resize", ParameterFlag::kOptional);
    spec.param(antialias_,
               "antialias",
               "apply antialiasing",
               "Whether to apply antialiasing when downsampling",
               true);
    spec.param(
        interpolation_,
        "interpolation",
        "interpolation",
        "Interpolation mode to use (used for both down and upsampling). If this is specified, "
        "min_interpolation and mag_interpolation should not be specified.",
        ParameterFlag::kOptional);
    spec.param(mag_interpolation_,
               "mag_interpolation",
               "mag_interpolation",
               "Interpolation mode to use for upsampling.",
               ParameterFlag::kOptional);
    spec.param(min_interpolation_,
               "min_interpolation_",
               "min_interpolation_",
               "Interpolation mode to use for downsampling",
               ParameterFlag::kOptional);
  }

  void initialize() {
    Operator::initialize();

    auto& current_args = args();

    auto interpolation_it =
        std::find_if(current_args.begin(), current_args.end(), [](const auto& arg) {
          return (arg.name() == "interpolation");
        });
    bool has_interpolation = interpolation_it != current_args.end();

    auto min_interpolation_it =
        std::find_if(current_args.begin(), current_args.end(), [](const auto& arg) {
          return (arg.name() == "min_interpolation");
        });
    bool has_min_interpolation = min_interpolation_it != current_args.end();

    auto mag_interpolation_it =
        std::find_if(current_args.begin(), current_args.end(), [](const auto& arg) {
          return (arg.name() == "mag_interpolation");
        });
    bool has_mag_interpolation = mag_interpolation_it != current_args.end();

    if (has_min_interpolation || has_mag_interpolation) {
      if (!has_min_interpolation) {
        throw std::runtime_error(
            "min_interpolation must be defined when mag_interpolation is defined.");
      }
      if (!has_mag_interpolation) {
        throw std::runtime_error(
            "mag_interpolation must be defined when min_interpolation is defined.");
      }
      if (has_interpolation) {
        HOLOSCAN_LOG_WARN(
            "interpolation parameter is ignored when both min_interpolation "
            "and mag_interpolation are defined.");
      }
    } else if (!has_interpolation) {
      throw std::runtime_error("no interpolation parameter was defined.");
    }

    auto roi_it = std::find_if(current_args.begin(), current_args.end(), [](const auto& arg) {
      return (arg.name() == "roi");
    });
    bool has_roi = roi_it != current_args.end();

    auto out_tensor_format_it =
        std::find_if(current_args.begin(), current_args.end(), [](const auto& arg) {
          return (arg.name() == "out_tensor_format");
        });
    bool has_out_tensor_format = out_tensor_format_it != current_args.end();
  }

  void start() {
    check_shape_and_format();
    check_roi();
    set_min_and_mag_interpolation();
  }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override {
    auto cv_in_tensor = op_input.receive<nvcv::Tensor>("input_tensor").value();

    auto ndim_expected = in_tensor_format_.get().size();
    if (cv_in_tensor.rank() != ndim_expected) {
      throw std::runtime_error(
          fmt::format("Tensor has {} dimensions, but expected a tensor with {} dimensions "
                      "corresponding to {} format",
                      cv_in_tensor.rank(),
                      ndim_expected,
                      in_tensor_format_.get()));
    }

    // determine number of channels from input tensor
    int32_t num_channels = in_index_c_ == -1 ? 1 : cv_in_tensor.shape()[in_index_c_];
    // determine batch size from the input tensor
    int32_t num_batch = in_index_n_ == -1 ? 1 : cv_in_tensor.shape()[in_index_n_];

    auto antialias = antialias_.get();

    // apply the HQResize operator
    cvcuda::HQResize resizeOp;

    // allocate the output tensor
    if (!out_tensor_allocated_) {
      int32_t height = 0;
      int32_t width = 0;
      int32_t depth = 0;
      // determine output height and width from input tensor or specified rect argument
      if (ndim_spatial_ == 2) {
        height = shape_.get()[0];
        width = shape_.get()[1];
      } else {
        depth = shape_.get()[0];
        height = shape_.get()[1];
        width = shape_.get()[2];
      }

      // output tensor with same num_batch and num_channels as the input
      // auto& in_fmt = in_tensor_format_.get();
      // auto& out_shape = out_shape._value();
      // out_shape.reserve(cv_in_tensor.rank());
      // if ((in_fmt == "HW") || (in_fmt == "DHW")) {
      //   for (auto& s : shape) {
      //     out_shape.push_back(s);
      //   }
      // } else if ((in_fmt == "HWC") || (in_fmt == "DHWC")) {
      //   for (auto& s : shape) {
      //     out_shape.push_back(s);
      //   }
      //   out_shape.push_back(num_channels);
      // } else if ((in_fmt == "NHWC") || (in_fmt == "NDHWC")) {
      //   out_shape.push_back(num_batch);
      //   for (auto& s : shape) {
      //     out_shape.push_back(s);
      //   }
      //   out_shape.push_back(num_channels);
      // }

      // allocate out_tensor_ of correct size
      auto in_dtype = cv_in_tensor.dtype();
      if (ndim_spatial_ == 2) {
        cv_out_tensor_ = CreateTensorHelper(
            cv_in_tensor.dtype(), "NHWC", num_batch, height, width, num_channels);
      } else {
        cv_out_tensor_ = CreateTensorHelper(
            cv_in_tensor.dtype(), "NDHWC", num_batch, depth, height, width, num_channels);
      }

      // allocate tensor memory
      auto out_data = cv_out_tensor_.exportData<nvcv::TensorDataStridedCuda>();
      auto out_reqs =
          nvcv::Tensor::CalcRequirements(cv_out_tensor_.shape(), cv_out_tensor_.dtype());
      size_t out_nbytes = nvcv::CalcTotalSizeBytes(nvcv::Requirements{out_reqs.mem}.cudaMem());

      // allocate the memory in the out buffer's basePtr
      auto data_ptr = out_data->basePtr();
      CUDA_TRY(cudaMalloc(&data_ptr, out_nbytes));

      // create workspace
      if (ndim_spatial_ == 2) {
        HQResizeTensorShapeI inShapeDesc{{height, width}, 2, num_channels};
        HQResizeTensorShapeI outShapeDesc{{height, width}, 2, num_channels};
        ws_ = cvcuda::AllocateWorkspace(resizeOp.getWorkspaceRequirements(num_batch,
                                                                          inShapeDesc,
                                                                          outShapeDesc,
                                                                          min_interpolation_enum_,
                                                                          mag_interpolation_enum_,
                                                                          antialias,
                                                                          hq_roi_ptr_));
      } else {
        HQResizeTensorShapeI inShapeDesc{{depth, height, width}, 3, num_channels};
        HQResizeTensorShapeI outShapeDesc{{depth, height, width}, 3, num_channels};
        ws_ = cvcuda::AllocateWorkspace(resizeOp.getWorkspaceRequirements(num_batch,
                                                                          inShapeDesc,
                                                                          outShapeDesc,
                                                                          min_interpolation_enum_,
                                                                          mag_interpolation_enum_,
                                                                          antialias,
                                                                          hq_roi_ptr_));
      }
      out_tensor_allocated_ = true;
    }

    cudaStream_t stream = cudaStreamDefault;  // TODO: use operator's internal stream instead
    resizeOp(stream,
             ws_.get(),
             cv_in_tensor,
             cv_out_tensor_,
             min_interpolation_enum_,
             mag_interpolation_enum_,
             antialias,
             hq_roi_ptr_);
    HOLOSCAN_LOG_DEBUG("HQResize done");

    // TODO (grelee): drop singleton 'N' and/or 'C' dimensions if requested

    // Emit the tensor
    op_output.emit(cv_out_tensor_, "output_tensor");
  }

 private:
  NVCVInterpolationType interp_name_to_enum(const std::string& value) {
    if (value == "nearest") {
      return NVCV_INTERP_NEAREST;
    } else if (value == "linear") {
      return NVCV_INTERP_LINEAR;
    } else if (value == "cubic") {
      return NVCV_INTERP_CUBIC;
    } else if (value == "lanczos") {
      return NVCV_INTERP_LANCZOS;
    } else if (value == "gaussian") {
      return NVCV_INTERP_GAUSSIAN;
    } else {
      throw std::runtime_error(
          "interpolation must be one of 'nearest', 'linear', 'cubic', 'lanczos' or 'gaussian'");
    }
  }

  void set_min_and_mag_interpolation() {
    if (min_interpolation_.has_value() && mag_interpolation_.has_value()) {
      min_interpolation_enum_ = interp_name_to_enum(min_interpolation_.get());
      mag_interpolation_enum_ = interp_name_to_enum(mag_interpolation_.get());
    } else if (interpolation_.has_value()) {
      min_interpolation_enum_ = interp_name_to_enum(interpolation_.get());
      mag_interpolation_enum_ = min_interpolation_enum_;
    } else {
      throw std::runtime_error(
          "either 'interpolation' or both 'min_interpolation' and 'mag_interpolation' must be "
          "defined.");
    }
  }

  void check_shape_and_format() {
    if (shape_.has_value()) {
      ndim_spatial_ = shape_.get().size();
      if ((ndim_spatial_ != 2) && (ndim_spatial_ != 3)) {
        throw std::runtime_error("shape.size() must be 2 for 2D images or 3 for 3D volumes");
      }
    }
    if (out_tensor_format_.has_value()) {
      auto out_fmt = out_tensor_format_.get();
      size_t d_pos = out_fmt.find("D");
      if ((ndim_spatial_ == 3) && d_pos == std::string::npos) {
        throw std::runtime_error(
            "expected character 'D' to be present in out_tensor_format when "
            "shape.size() == 3.");
      }
    } else {
      out_tensor_format_ = in_tensor_format_.get();
    }
    if (in_tensor_format_.has_value()) {
      auto in_fmt = in_tensor_format_.get();
      size_t found = in_fmt.find("N");
      if (found != std::string::npos) { in_index_n_ = found; }
      found = in_fmt.find("C");
      if (found != std::string::npos) { in_index_c_ = found; }
      found = in_fmt.find("H");
      if (found != std::string::npos) { in_index_h_ = found; }
      found = in_fmt.find("W");
      if (found != std::string::npos) { in_index_w_ = found; }
    } else {
      throw std::runtime_error("No value found for required in_tensor_format argument");
    }

    // TODO: validate that format is one of "HW", "HWC", "NHWC", "DHW", "DHWC", "NDHWC"
  }

  void check_roi() {
    if (roi_.has_value()) {
      // convert ROI vector to HQResizeRoiF* struct pointer
      auto roi = roi_.get();
      if (roi.size() > 0) {
        if (ndim_spatial_ == 2) {
          if (roi.size() != 4) {
            throw std::runtime_error(
                "For 2D images, the ROI must have size 4: "
                "(h_low, w_low, h_high, w_high)");
          }
          hq_roi_ptr_->lo[0] = roi[0];
          hq_roi_ptr_->lo[1] = roi[1];
          hq_roi_ptr_->hi[0] = roi[2];
          hq_roi_ptr_->hi[1] = roi[3];
        } else {
          if (roi.size() != 6) {
            throw std::runtime_error(
                "For 3D images, the ROI must have size 6: "
                "(d_low, h_low, w_low, d_low, h_high, w_high)");
          }
          hq_roi_ptr_->lo[0] = roi[0];
          hq_roi_ptr_->lo[1] = roi[1];
          hq_roi_ptr_->lo[3] = roi[2];
          hq_roi_ptr_->hi[0] = roi[3];
          hq_roi_ptr_->hi[1] = roi[4];
          hq_roi_ptr_->hi[2] = roi[5];
        }
      }
    }
  }

  Parameter<std::vector<int32_t>> shape_;     // shape (length 2 or 3)
  Parameter<std::string> in_tensor_format_;   // The format of the input tensor ("NHWC, etc.")
  Parameter<std::string> out_tensor_format_;  // The format of the output tensor ("NHWC, etc.")
  Parameter<std::vector<int32_t>> roi_;  // region-of-interest: length 4 for 2D, length 6 for 3D
  Parameter<bool> antialias_;            // whether antialiasing is performed when downsizing
  // support string or enum type for Interpolation?
  Parameter<std::string> interpolation_;      // The interpolation type to use
  Parameter<std::string> min_interpolation_;  // The interpolation type to use for downscaling
  Parameter<std::string> mag_interpolation_;  // The interpolation type to use for upscaling
  NVCVInterpolationType min_interpolation_enum_ = NVCV_INTERP_LINEAR;
  NVCVInterpolationType mag_interpolation_enum_ = NVCV_INTERP_LINEAR;
  int32_t ndim_spatial_ = 0;
  int32_t in_index_n_ = -1;
  int32_t in_index_c_ = -1;
  int32_t in_index_h_ = -1;
  int32_t in_index_w_ = -1;
  bool out_tensor_allocated_ = false;
  nvcv::Tensor cv_out_tensor_;
  cvcuda::UniqueWorkspace ws_;
  HQResizeRoiF* hq_roi_ptr_ = nullptr;
};

}  // namespace holoscan::ops

class App : public holoscan::Application {
 public:
  void set_datapath(const std::string& path) { datapath = path; }

  void compose() override {
    using namespace holoscan;

    const std::shared_ptr<CudaStreamPool> cuda_stream_pool =
        make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5);

    uint32_t width = 854;
    uint32_t height = 480;
    auto source = make_operator<ops::VideoStreamReplayerOp>(
        "replayer", from_config("replayer"), Arg("directory", datapath));

    auto holoscan_to_cvcuda = make_operator<ops::HoloscanToCvCuda>("holoscan_to_cvcuda");

    auto image_processing =
        make_operator<ops::HQResizeOp>("image_processing",
                                       Arg("shape", std::vector<int32_t>({720, 1280})),
                                       Arg("in_tensor_format", std::string("NHWC")));

    auto cvcuda_to_holoscan = make_operator<ops::CvCudaToHoloscan>("cvcuda_to_holoscan");

    std::shared_ptr<ops::HolovizOp> visualizer1 =
        make_operator<ops::HolovizOp>("holoviz1",
                                      from_config("holoviz1"),
                                      Arg("window_title") = std::string("Original Stream"),
                                      Arg("width") = width,
                                      Arg("height") = height,
                                      Arg("cuda_stream_pool") = cuda_stream_pool);

    std::shared_ptr<ops::HolovizOp> visualizer2 =
        make_operator<ops::HolovizOp>("holoviz2",
                                      from_config("holoviz2"),
                                      Arg("window_title") = std::string("Flipped Stream"),
                                      Arg("width") = width,
                                      Arg("height") = height,
                                      Arg("cuda_stream_pool") = cuda_stream_pool);

    // Flow definition
    // add_flow(source, visualizer1, {{"output", "receivers"}}); // optional to watch the original
    // stream
    add_flow(source, holoscan_to_cvcuda);
    add_flow(holoscan_to_cvcuda, image_processing);
    add_flow(image_processing, cvcuda_to_holoscan);
    add_flow(cvcuda_to_holoscan, visualizer2, {{"output", "receivers"}});
  }

 private:
  std::string datapath = "data/endoscopy";
};

/** Helper function to parse the command line arguments */
bool parse_arguments(int argc, char** argv, std::string& config_name, std::string& data_path) {
  static struct option long_options[] = {{"data", required_argument, 0, 'd'}, {0, 0, 0, 0}};

  while (int c = getopt_long(argc, argv, "d", long_options, NULL)) {
    if (c == -1 || c == '?') break;

    switch (c) {
      case 'd':
        data_path = optarg;
        break;
      default:
        std::cout << "Unknown arguments returned: " << c << std::endl;
        return false;
    }
  }

  if (optind < argc) { config_name = argv[optind++]; }
  return true;
}

/** Main function */
int main(int argc, char** argv) {
  auto app = holoscan::make_application<App>();

  // Parse the arguments
  std::string data_path = "";
  std::string config_name = "";
  if (!parse_arguments(argc, argv, config_name, data_path)) { return 1; }

  if (config_name != "") {
    app->config(config_name);
  } else {
    auto config_path = std::filesystem::canonical(argv[0]).parent_path();
    config_path += "/cvcuda_basic.yaml";
    app->config(config_path);
  }
  if (data_path != "") app->set_datapath(data_path);

  app->run();

  return 0;
}
