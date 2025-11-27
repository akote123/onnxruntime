/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    erf.cpp

Abstract:

    This module implements routines to compute the hyperbolic tangent function.

    This implementation uses the same polynomial coefficients and algorithm as
    found in: https://stackoverflow.com/questions/35148198/efficient-faithfully-rounded-implementation-of-error-function-erff
    Our usage requires building platform specific versions of
    the algorithm to target different instruction sets. The implementation below
    targets the base instruction set (typically SSE2) while assembly
    implementations target newer instruction sets (such as FMA3).

--*/

#include "mlasi.h"
#include "fp16_common.h"
#include "softmax_kernel_neon.h"

using _mlas_fp16_ = uint16_t;
//
// Bundles the constants for use by kernels written in assembly.
//

MLAS_INTERNAL_DATA const struct {
    float ErfUpperAbsRange;
    float ErfSplitBoundary;
    float ErfSMALL_P0;
    float ErfSMALL_P1;
    float ErfSMALL_P2;
    float ErfSMALL_P3;
    float ErfSMALL_P4;
    float ErfSMALL_P5_Minus_One;
    float ErfReserved0;
    float ErfBIG_P0;
    float ErfBIG_P1;
    float ErfBIG_P2;
    float ErfBIG_P3;
    float ErfBIG_P4;
    float ErfBIG_P5;
    float ErfBIG_P6_Minus_One;
    float ErfNegZero;
    float ErfOne;

    float Exp_UpperRange;
    float Exp_LowerRange;
    float Exp_Log2Reciprocal;
    float Exp_log2_hi;
    float Exp_log2_lo;
    float Exp_P0;
    float Exp_P1;
    float Exp_P2;
    float Exp_P3;
    float Exp_P4;
    float Exp_P5;
    float Exp_P6;
    float Exp_C;
    int32_t Exp_X7F;
} MlasErfConstants = {
    3.925f,
    0.921875f,
    -5.99104969e-4f,
    4.99339588e-3f,
    -2.67667342e-2f,
    1.12818025e-1f,
    -3.76124859e-1f,
    1.28379151e-1f,
    0.0f,
    1.72948930e-5f,
    -3.83208680e-4f,
    3.88393435e-3f,
    -2.42545605e-2f,
    1.06777847e-1f,
    6.34846687e-1f,
    1.28717512e-1f,
    -0.0f,
    1.0f,

    // Independent parameters to calculate Exp for Erff()
    88.3762626647950f,
    -88.3762626647949f,
    1.44269504088896341f,
    -6.93145752e-1f,
    -1.42860677e-6f,
    1.38319808e-3f,
    8.37550033e-3f,
    4.16689515e-2f,
    1.66664466e-1f,
    4.99999851e-1f,
    1.00000000e+0f,
    1.00000000e+0f,
    1.25829120e+7f,
    127,
};

void
MLASCALL
MlasErfKernel(
    const float* Input,
    float* Output,
    size_t N
    )
/*++

Routine Description:

    This routine implements the generic kernel for the error function.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

Return Value:

    None.

--*/
{
    while (N >= 4) {
        MLAS_FLOAT32X4 Value = MlasLoadFloat32x4(Input);
        MLAS_FLOAT32X4 NegZero = MlasBroadcastFloat32x4(MlasErfConstants.ErfNegZero);
        MLAS_FLOAT32X4 SignMask = MlasAndFloat32x4(Value, NegZero);
        MLAS_FLOAT32X4 AbsValue = MlasAndNotFloat32x4(NegZero, Value);
        AbsValue = MlasMinimumFloat32x4(MlasBroadcastFloat32x4(MlasErfConstants.ErfUpperAbsRange), AbsValue);
        MLAS_FLOAT32X4 SquareValue = MlasMultiplyFloat32x4(AbsValue, AbsValue);

        MLAS_FLOAT32X4 r_small = MlasBroadcastFloat32x4(MlasErfConstants.ErfSMALL_P0);
        r_small = MlasMultiplyAddFloat32x4(r_small, SquareValue, MlasBroadcastFloat32x4(MlasErfConstants.ErfSMALL_P1));
        r_small = MlasMultiplyAddFloat32x4(r_small, SquareValue, MlasBroadcastFloat32x4(MlasErfConstants.ErfSMALL_P2));
        r_small = MlasMultiplyAddFloat32x4(r_small, SquareValue, MlasBroadcastFloat32x4(MlasErfConstants.ErfSMALL_P3));
        r_small = MlasMultiplyAddFloat32x4(r_small, SquareValue, MlasBroadcastFloat32x4(MlasErfConstants.ErfSMALL_P4));
        r_small = MlasMultiplyAddFloat32x4(r_small, SquareValue, MlasBroadcastFloat32x4(MlasErfConstants.ErfSMALL_P5_Minus_One));
        r_small = MlasMultiplyAddFloat32x4(r_small, AbsValue, AbsValue);
        MLAS_FLOAT32X4 split_mask = MlasGreaterThanFloat32x4(AbsValue, MlasBroadcastFloat32x4(MlasErfConstants.ErfSplitBoundary));
        r_small = MlasAndNotFloat32x4(split_mask, r_small);

        AbsValue = MlasAndFloat32x4(split_mask, AbsValue); // clear smaller value into zero for bigger number calculation
        MLAS_FLOAT32X4 r_big = MlasBroadcastFloat32x4(MlasErfConstants.ErfBIG_P0);
        r_big = MlasMultiplyAddFloat32x4(r_big, AbsValue, MlasBroadcastFloat32x4(MlasErfConstants.ErfBIG_P1));
        r_big = MlasMultiplyAddFloat32x4(r_big, AbsValue, MlasBroadcastFloat32x4(MlasErfConstants.ErfBIG_P2));
        r_big = MlasMultiplyAddFloat32x4(r_big, AbsValue, MlasBroadcastFloat32x4(MlasErfConstants.ErfBIG_P3));
        r_big = MlasMultiplyAddFloat32x4(r_big, AbsValue, MlasBroadcastFloat32x4(MlasErfConstants.ErfBIG_P4));
        r_big = MlasMultiplyAddFloat32x4(r_big, AbsValue, MlasBroadcastFloat32x4(MlasErfConstants.ErfBIG_P5));
        r_big = MlasMultiplyAddFloat32x4(r_big, AbsValue, MlasBroadcastFloat32x4(MlasErfConstants.ErfBIG_P6_Minus_One));
        r_big = MlasMultiplyAddFloat32x4(r_big, AbsValue, AbsValue);

        // 1.0 - exp(-r_big), no need to do min()
        r_big = MlasXorFloat32x4(r_big, MlasBroadcastFloat32x4(MlasErfConstants.ErfNegZero)); // -r_big
        r_big = MlasMaximumFloat32x4(MlasBroadcastFloat32x4(MlasErfConstants.Exp_LowerRange), r_big);
        MLAS_FLOAT32X4 exp_c = MlasBroadcastFloat32x4(MlasErfConstants.Exp_C);
        MLAS_FLOAT32X4 r = MlasMultiplyAddFloat32x4(MlasBroadcastFloat32x4(MlasErfConstants.Exp_Log2Reciprocal), r_big, exp_c);
        r = MlasSubtractFloat32x4(r, exp_c);

        MLAS_FLOAT32X4 fx = MlasMultiplyAddFloat32x4(r, MlasBroadcastFloat32x4(MlasErfConstants.Exp_log2_hi), r_big);
        fx = MlasMultiplyAddFloat32x4(r, MlasBroadcastFloat32x4(MlasErfConstants.Exp_log2_lo), fx);
        // y = exp(fx)
        MLAS_FLOAT32X4 y = MlasBroadcastFloat32x4(MlasErfConstants.Exp_P0);
        y = MlasMultiplyAddFloat32x4(y, fx, MlasBroadcastFloat32x4(MlasErfConstants.Exp_P1));
        y = MlasMultiplyAddFloat32x4(y, fx, MlasBroadcastFloat32x4(MlasErfConstants.Exp_P2));
        y = MlasMultiplyAddFloat32x4(y, fx, MlasBroadcastFloat32x4(MlasErfConstants.Exp_P3));
        y = MlasMultiplyAddFloat32x4(y, fx, MlasBroadcastFloat32x4(MlasErfConstants.Exp_P4));
        y = MlasMultiplyAddFloat32x4(y, fx, MlasBroadcastFloat32x4(MlasErfConstants.Exp_P5));
        y = MlasMultiplyAddFloat32x4(y, fx, MlasBroadcastFloat32x4(MlasErfConstants.Exp_P6));
        // 1.0 - exp(fx) * 2^INT(r)
        y = MlasMultiplyFloat32x4(y, MlasPowerOf2Float32x4(r));
        y = MlasSubtractFloat32x4(MlasBroadcastFloat32x4(MlasErfConstants.ErfOne), y);

        // merge two splits results
        y = MlasOrFloat32x4(r_small, y);
        y = MlasOrFloat32x4(y, SignMask);

        MlasStoreFloat32x4(Output, y);

        Input += 4;
        Output += 4;
        N -= 4;
    }

    while (N > 0) {
        float Value = *Input++;
        float AbsValue = fabsf(Value);

        float r;
        if (AbsValue > MlasErfConstants.ErfSplitBoundary) {
            AbsValue = std::min(MlasErfConstants.ErfUpperAbsRange, AbsValue);
            float r_big = MlasErfConstants.ErfBIG_P0;
            r_big = r_big * AbsValue + MlasErfConstants.ErfBIG_P1;
            r_big = r_big * AbsValue + MlasErfConstants.ErfBIG_P2;
            r_big = r_big * AbsValue + MlasErfConstants.ErfBIG_P3;
            r_big = r_big * AbsValue + MlasErfConstants.ErfBIG_P4;
            r_big = r_big * AbsValue + MlasErfConstants.ErfBIG_P5;
            r_big = r_big * AbsValue + MlasErfConstants.ErfBIG_P6_Minus_One;
            r_big = r_big * AbsValue + AbsValue;

            r_big = std::max(-r_big, MlasErfConstants.Exp_LowerRange);
            r = MlasErfConstants.Exp_Log2Reciprocal * r_big + MlasErfConstants.Exp_C;
            r -= MlasErfConstants.Exp_C;
            float fx = r * MlasErfConstants.Exp_log2_hi + r_big;
            fx = r * MlasErfConstants.Exp_log2_lo + fx;

            float y = MlasErfConstants.Exp_P0;
            y = y * fx + MlasErfConstants.Exp_P1;
            y = y * fx + MlasErfConstants.Exp_P2;
            y = y * fx + MlasErfConstants.Exp_P3;
            y = y * fx + MlasErfConstants.Exp_P4;
            y = y * fx + MlasErfConstants.Exp_P5;
            y = y * fx + MlasErfConstants.Exp_P6;

            r = 1.0f - ldexpf(y, (int)r);
            r = (Value <= -0.0f) ? -r : r;
        }
        else {
            float SquareValue = AbsValue * AbsValue;
            r = MlasErfConstants.ErfSMALL_P0;
            r = r * SquareValue + MlasErfConstants.ErfSMALL_P1;
            r = r * SquareValue + MlasErfConstants.ErfSMALL_P2;
            r = r * SquareValue + MlasErfConstants.ErfSMALL_P3;
            r = r * SquareValue + MlasErfConstants.ErfSMALL_P4;
            r = r * SquareValue + MlasErfConstants.ErfSMALL_P5_Minus_One;
            r = r * Value + Value;
        }

        *Output++ = r;
        N -= 1;
    }
}

void
MLASCALL
MlasComputeErf(
    const float* Input,
    float* Output,
    size_t N
    )
/*++

Routine Description:

    This routine computes the error function.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

Return Value:

    None.

--*/
{
#if defined(MLAS_TARGET_AMD64) || defined(MLAS_USE_SVE)
    GetMlasPlatform().ErfKernelRoutine(Input, Output, N);
#else
    MlasErfKernel(Input, Output, N);
#endif
}

// Helpers to safely convert between float and FP16-bit representation
static float
fp16_to_float(uint16_t h)
{
    __fp16 tmp;
    memcpy(&tmp, &h, sizeof(h));
    return (float)tmp;
}

static uint16_t
float_to_fp16(float f)
{
    __fp16 tmp = (__fp16)f;
    uint16_t h;
    memcpy(&h, &tmp, sizeof(h));
    return h;
}

static inline MLAS_FLOAT16X8
exp_neg_rational_approx_f16(MLAS_FLOAT16X8 x)
{
    const float16_t a0 = 6.0f;
    MLAS_FLOAT16X8 max_x = MlasBroadcastF16Float16x8(a0);
    x = MlasMinimumFloat16(x, max_x);

    const float16_t c0 = 1.330f;
    const float16_t c1 = -0.390f;
    const float16_t c2 = 0.0288f;

    const float16_t d0 = 1.338f;
    const float16_t d1 = 0.848f;
    const float16_t d2 = 0.467f;

    MLAS_FLOAT16X8 c0v = MlasBroadcastF16Float16x8(c0);
    MLAS_FLOAT16X8 c1v = MlasBroadcastF16Float16x8(c1);
    MLAS_FLOAT16X8 c2v = MlasBroadcastF16Float16x8(c2);
    MLAS_FLOAT16X8 d0v = MlasBroadcastF16Float16x8(d0);
    MLAS_FLOAT16X8 d1v = MlasBroadcastF16Float16x8(d1);
    MLAS_FLOAT16X8 d2v = MlasBroadcastF16Float16x8(d2);
    MLAS_FLOAT16X8 x2 = MlasMultiplyFloat16(x, x);
    MLAS_FLOAT16X8 num = MlasMultiplyAddFloat16(c1v, x,c0v);
    num = MlasMultiplyAddFloat16(c2v, x2,num);
    MLAS_FLOAT16X8 den = MlasMultiplyAddFloat16(d1v, x,d0v);
    den = MlasMultiplyAddFloat16(d2v, x2,den);
    MLAS_FLOAT16X8 recip = MlasapproximatereciprocalFloat16(den);
    recip = MlasMultiplyFloat16(recip, MlasreciprocalsqrtFloat16(den, recip));
    recip = MlasMultiplyFloat16(recip, MlasreciprocalsqrtFloat16(den, recip));

    MLAS_FLOAT16X8 result = MlasMultiplyFloat16(num, recip);
    return result;
}

void
MlasNeonErfKernelFp16(const _mlas_fp16_* Input, _mlas_fp16_* Output, size_t N)
{
    const float16_t p = 0.328f;
    const float16_t a1 = 0.2505f;
    const float16_t a2 = -0.2881f;
    const float16_t a3 = 1.4102f;
    const float16_t a4 = -1.423f;
    const float16_t a5 = 1.0547f;

    MLAS_FLOAT16X8 vp = MlasBroadcastF16Float16x8(p);
    MLAS_FLOAT16X8 va1 = MlasBroadcastF16Float16x8(a1);
    MLAS_FLOAT16X8 va2 = MlasBroadcastF16Float16x8(a2);
    MLAS_FLOAT16X8 va3 = MlasBroadcastF16Float16x8(a3);
    MLAS_FLOAT16X8 va4 = MlasBroadcastF16Float16x8(a4);
    MLAS_FLOAT16X8 va5 = MlasBroadcastF16Float16x8(a5);

    constexpr float16_t one_fp16 = 1.0f;
    constexpr float16_t neg_one_fp16 = -1.0f;
    constexpr float16_t zero_fp16 = 0.0f;
    constexpr float16_t four_fp16 = 4.0f;

    MLAS_FLOAT16X8 vone = MlasBroadcastF16Float16x8(one_fp16);
    MLAS_FLOAT16X8 vneg_one = MlasBroadcastF16Float16x8(neg_one_fp16);
    MLAS_FLOAT16X8 vzero = MlasBroadcastF16Float16x8(zero_fp16);
    MLAS_FLOAT16X8 vth = MlasBroadcastF16Float16x8(four_fp16);
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        MLAS_FLOAT16X8 x = MlasLoadFloat16x8(&Input[i]);
        MLAS_UINT16X8 neg_mask = MlasComparelessthanFloat16(x, vzero);
        MLAS_FLOAT16X8 sign = MlasselectFloat16(neg_mask, vneg_one, vone);
        MLAS_FLOAT16X8 absx = MlasAbsFloat16(x);
        MLAS_UINT16X8 use_mask = MlasComparelessthanFloat16(absx, vth);
        MLAS_FLOAT16X8 absx_clamped = MlasMinimumFloat16(absx, vth);
        MLAS_FLOAT16X8 denom = MlasMultiplyAddFloat16(vp, absx_clamped,vone);
        MLAS_FLOAT16X8 t = MlasapproximatereciprocalFloat16(denom);
        t = MlasMultiplyFloat16(t, MlasreciprocalsqrtFloat16(denom, t));
        t = MlasMultiplyFloat16(t, MlasreciprocalsqrtFloat16(denom, t));
        MLAS_FLOAT16X8 t2 = MlasMultiplyFloat16(t, t);
        MLAS_FLOAT16X8 t3 = MlasMultiplyFloat16(t2, t);
        MLAS_FLOAT16X8 t4 = MlasMultiplyFloat16(t3, t);
        MLAS_FLOAT16X8 t5 = MlasMultiplyFloat16(t4, t);
        MLAS_FLOAT16X8 poly = MlasMultiplyFloat16(va1, t);
        poly = MlasMultiplyAddFloat16(va2, t2,poly);
        poly = MlasMultiplyAddFloat16(va3, t3,poly);
        poly = MlasMultiplyAddFloat16(va4, t4,poly);
        poly = MlasMultiplyAddFloat16(va5, t5,poly);
        MLAS_FLOAT16X8 x2 = MlasMultiplyFloat16(absx_clamped, absx_clamped);
        MLAS_FLOAT16X8 exp_neg_x2 = exp_neg_rational_approx_f16(x2);
        MLAS_FLOAT16X8 poly_mul_exp = MlasMultiplyFloat16(poly, exp_neg_x2);
        MLAS_FLOAT16X8 one_minus_term = MlasSubtractFloat16(vone, poly_mul_exp);
        MLAS_FLOAT16X8 erf_approx = MlasMultiplyFloat16(sign, one_minus_term);
        erf_approx = MlasMinimumFloat16(erf_approx, vone);
        erf_approx = MlasMaximumFloat16(erf_approx, vneg_one);
        MLAS_FLOAT16X8 result = MlasselectFloat16(use_mask, erf_approx, sign);
        MlasStoreFloat16x8(&Output[i], result);
    }

    for (; i < N; i++) {
        float x = fp16_to_float(Input[i]);
        float sign = (x < 0) ? -1.0f : 1.0f;
        float absx = fabsf(x);

        if (absx > 4.0f) {
            Output[i] = float_to_fp16(sign);
            continue;
        }

        float t = 1.0f / (1.0f + p * absx);
        float poly = a1 * t + a2 * t * t + a3 * t * t * t + a4 * t * t * t * t + a5 * t * t * t * t * t;
        float exp_neg_x2 = expf(-absx * absx);
        float erf_approx = sign * (1.0f - poly * exp_neg_x2);
        if (erf_approx > 1.0f) erf_approx = 1.0f;
        if (erf_approx < -1.0f) erf_approx = -1.0f;

        Output[i] = float_to_fp16(erf_approx);
    }
}

