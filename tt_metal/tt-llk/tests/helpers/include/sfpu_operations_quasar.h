// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_math_eltwise_unary_sfpu.h"

// To add a new Quasar unary SFPU operation:
// 1. Include its `ckernel_sfpu_<op>.h` below.
// 2. Add the `SfpuType` enumerator to the `if constexpr` chain in
//    call_unary_sfpu_operation_quasar() (and to init_unary_sfpu_operation_quasar()
//    if the op needs an init step).
#include "experimental/ckernel_sfpu_abs.h"
#include "llk_sfpu/ckernel_sfpu_comp.h"
#include "sfpu/ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_gelu.h"
#include "sfpu/ckernel_sfpu_recip.h"
#include "sfpu/ckernel_sfpu_relu.h"
#include "sfpu/ckernel_sfpu_rsqrt.h"
#include "sfpu/ckernel_sfpu_sigmoid.h"
#include "sfpu/ckernel_sfpu_silu.h"
#include "sfpu/ckernel_sfpu_sqrt.h"
#include "sfpu/ckernel_sfpu_square.h"
#include "sfpu/ckernel_sfpu_tanh.h"

namespace test_utils
{
using namespace ckernel;
using namespace ckernel::sfpu;

/**
 * @brief Whether OPERATION is one of the six comparison-to-zero modes.
 *
 * The comp family needs a dedicated init (@ref _init_zero_comp_) and a runtime
 * format switch (@ref call_zero_comp_operation_quasar), unlike the float-only
 * unary ops, so the dispatchers special-case it.
 *
 * @param op The SFPU operation type to classify.
 */
inline constexpr bool is_zero_comp_op(SfpuType op)
{
    return op == SfpuType::equal_zero || op == SfpuType::not_equal_zero || op == SfpuType::less_than_zero || op == SfpuType::greater_than_zero ||
           op == SfpuType::less_than_equal_zero || op == SfpuType::greater_than_equal_zero;
}

/**
 * @brief Run the per-operation init step for a Quasar unary SFPU op.
 *
 * Most Quasar unary ops need no dedicated init (the shared
 * `_llk_math_eltwise_sfpu_init_()` in the kernel is sufficient); only the ops
 * with a `_init_*_` entry point are handled here. This mirrors the inline
 * dispatcher previously embedded in `sfpu_nonlinear_quasar_test.cpp`, which
 * initialised only gelu.
 *
 * @tparam OPERATION The SFPU operation type (compile-time `SfpuType` constant).
 * @note Pair with @ref call_unary_sfpu_operation_quasar for the calculate step.
 */
template <SfpuType OPERATION>
void init_unary_sfpu_operation_quasar()
{
    if constexpr (OPERATION == SfpuType::gelu)
    {
        _init_gelu_();
    }
    else if constexpr (is_zero_comp_op(OPERATION))
    {
        _init_zero_comp_();
    }
}

/**
 * @brief Apply a comparison-to-zero SFPU op in-place on one Dest tile.
 *
 * Unlike the float-only unary ops, comp needs the SFPU math format at runtime to
 * pick the sfpmem load/store mode and the 1/0 result encoding (see
 * `ckernel_sfpu_comp.h`). Integer formats select their explicit width; all float
 * widths share the width-agnostic `Float32` instantiation (SFPLOAD/SFPSTORE
 * DEFAULT resolves the actual width from the HW format config).
 *
 * @tparam OPERATION The comparison-to-zero `SfpuType` (compile-time constant).
 * @tparam ITERATIONS Number of SFPU loop iterations.
 * @param dst_index Destination tile index operated on (already offset by DST_INDEX).
 * @param sfpu_format SFPU math format selecting the sfpmem mode / result encoding.
 * @note Must be preceded by @ref init_unary_sfpu_operation_quasar for the same op.
 */
template <SfpuType OPERATION, int ITERATIONS = SFPU_ITERATIONS>
void call_zero_comp_operation_quasar(std::uint32_t dst_index, DataFormat sfpu_format)
{
    switch (sfpu_format)
    {
        case DataFormat::Int32:
            _llk_math_eltwise_unary_sfpu_params_(_calculate_zero_comp_<false, DataFormat::Int32, OPERATION, ITERATIONS>, dst_index);
            break;
        case DataFormat::Int16:
            _llk_math_eltwise_unary_sfpu_params_(_calculate_zero_comp_<false, DataFormat::Int16, OPERATION, ITERATIONS>, dst_index);
            break;
        case DataFormat::UInt16:
            _llk_math_eltwise_unary_sfpu_params_(_calculate_zero_comp_<false, DataFormat::UInt16, OPERATION, ITERATIONS>, dst_index);
            break;
        default:
            // Float16 / Float16_b / Float32 — width-agnostic float path.
            _llk_math_eltwise_unary_sfpu_params_(_calculate_zero_comp_<false, DataFormat::Float32, OPERATION, ITERATIONS>, dst_index);
            break;
    }
}

/**
 * @brief Apply a Quasar unary SFPU op in-place on one Dest tile.
 *
 * @tparam OPERATION The SFPU operation type (compile-time `SfpuType` constant).
 * @tparam ITERATIONS Number of SFPU loop iterations.
 * @param dst_index Destination tile index operated on (already offset by DST_INDEX).
 * @param sfpu_format SFPU math format; only the comp family reads it (see
 *        @ref call_zero_comp_operation_quasar), float-only ops ignore it.
 * @note Must be preceded by @ref init_unary_sfpu_operation_quasar for the same op.
 */
template <SfpuType OPERATION, int ITERATIONS = SFPU_ITERATIONS>
void call_unary_sfpu_operation_quasar(std::uint32_t dst_index, DataFormat sfpu_format = DataFormat::Float32)
{
    if constexpr (OPERATION == SfpuType::abs)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_abs_<ITERATIONS>, dst_index);
    }
    else if constexpr (OPERATION == SfpuType::exponential)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_exp_<true /* APPROX */, ITERATIONS>, dst_index);
    }
    else if constexpr (OPERATION == SfpuType::gelu)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_gelu_<ITERATIONS>, dst_index);
    }
    else if constexpr (OPERATION == SfpuType::relu)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_relu_<ITERATIONS>, dst_index);
    }
    else if constexpr (OPERATION == SfpuType::reciprocal)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_reciprocal_<true /* APPROX */, ITERATIONS>, dst_index);
    }
    else if constexpr (OPERATION == SfpuType::sqrt)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_sqrt_<true /* APPROX */, ITERATIONS>, dst_index);
    }
    else if constexpr (OPERATION == SfpuType::tanh)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_tanh_<true /* APPROX */, ITERATIONS>, dst_index);
    }
    else if constexpr (OPERATION == SfpuType::sigmoid)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_sigmoid_<ITERATIONS>, dst_index);
    }
    else if constexpr (OPERATION == SfpuType::silu)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_silu_<ITERATIONS>, dst_index);
    }
    else if constexpr (OPERATION == SfpuType::rsqrt)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_rsqrt_<ITERATIONS>, dst_index);
    }
    else if constexpr (OPERATION == SfpuType::square)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_square_<ITERATIONS>, dst_index);
    }
    else if constexpr (is_zero_comp_op(OPERATION))
    {
        call_zero_comp_operation_quasar<OPERATION, ITERATIONS>(dst_index, sfpu_format);
    }
    else
    {
        LLK_ASSERT(false, "Unsupported Quasar unary SFPU operation");
    }
}

} // namespace test_utils
