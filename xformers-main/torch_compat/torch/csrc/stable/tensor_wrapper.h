#pragma once

// Wrapper for torch/csrc/stable/tensor.h that avoids including tensor_inl.h
// to prevent multiple definition errors for scalar_type() when included in multiple .cu files
//
// The scalar_type() function is declared in tensor_struct.h and implemented in tensor_wrapper.cpp
// This ensures the function is only defined once, even when included in multiple translation units

#include <torch/csrc/stable/tensor_struct.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/shim_utils.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

// Now include the rest of tensor.h functionality
// We skip tensor_inl.h to avoid the non-inline scalar_type() definition
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>

