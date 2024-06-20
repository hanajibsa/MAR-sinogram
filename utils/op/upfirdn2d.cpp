// ---------------------------------------------------------------
// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
// ---------------------------------------------------------------

// Originated from https://github.com/rosinality/stylegan2-pytorch
// The license for the original version of this file can be found in this directory (LICENSE_MIT).


#include <torch/extension.h>


torch::Tensor upfirdn2d_op(const torch::Tensor& input, const torch::Tensor& kernel,
                            int up_x, int up_y, int down_x, int down_y,
                            int pad_x0, int pad_x1, int pad_y0, int pad_y1);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor upfirdn2d(const torch::Tensor& input, const torch::Tensor& kernel,
                        int up_x, int up_y, int down_x, int down_y,
                        int pad_x0, int pad_x1, int pad_y0, int pad_y1) {
    CHECK_CUDA(input);
    CHECK_CUDA(kernel);

    return upfirdn2d_op(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("upfirdn2d", &upfirdn2d, "upfirdn2d (CUDA)");
}

// # import torch

// # # Load the JIT-compiled C++ extension
// # upfirdn2d_op = torch.ops.load_library("path/to/your/libupfirdn2d.so")

// # # Define a wrapper function to call the C++ function
// # def upfirdn2d(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
// #     return upfirdn2d_op.upfirdn2d(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1)

// # # Example usage
// # input_tensor = torch.randn(1, 3, 64, 64)  # Example input tensor
// # kernel_tensor = torch.randn(3, 3, 3, 3)    # Example kernel tensor
// # result = upfirdn2d(input_tensor, kernel_tensor, 2, 2, 2, 2, 1, 1, 1, 1)
