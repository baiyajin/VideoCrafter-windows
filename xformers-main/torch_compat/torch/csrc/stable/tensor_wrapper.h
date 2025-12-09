#pragma once

// Wrapper for torch/csrc/stable/tensor.h that provides inline scalar_type()
// to avoid multiple definition errors when included in multiple .cu files

#include <torch/csrc/stable/tensor_struct.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/shim_utils.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

namespace torch::stable {
using torch::headeronly::ScalarType;

// Declaration only - implementation is in tensor_wrapper.cpp to avoid multiple definition errors
// This ensures the function is only defined once, even when included in multiple translation units
ScalarType Tensor::scalar_type() const;
} // namespace torch::stable

// Now include the rest of tensor.h functionality
// We skip tensor_inl.h to avoid the non-inline scalar_type() definition
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>

