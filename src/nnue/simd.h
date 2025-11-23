#pragma once

#include "../defs.h"

#ifndef ARM
#include <immintrin.h>
#else
#include <arm_neon.h>
#endif

namespace Horsie::NNUE {

#ifndef ARM
    using vec_128i = __m128i;
#else
    using vec_128i = int16x8_t;
#endif


#if defined(AVX512)

    using vec_i8 = __m512i;
    using vec_i16 = __m512i;
    using vec_i32 = __m512i;
    using vec_ps = __m512;

    inline vec_i8 vec_packus_epi16(const vec_i16 a, const vec_i16 b) { return _mm512_packus_epi16(a, b); }
    inline void vec_storeu_epi8(vec_i8* a, const vec_i8 b) { _mm512_storeu_si512(a, b); }

    inline vec_ps vec_set1_ps(const float a) { return _mm512_set1_ps(a); }
    inline vec_ps vec_fmadd_ps(const vec_ps a, const vec_ps b, const vec_ps c) { return _mm512_fmadd_ps(a, b, c); }
    inline vec_ps vec_min_ps(const vec_ps a, const vec_ps b) { return _mm512_min_ps(a, b); }
    inline vec_ps vec_max_ps(const vec_ps a, const vec_ps b) { return _mm512_max_ps(a, b); }
    inline vec_ps vec_mul_ps(const vec_ps a, const vec_ps b) { return _mm512_mul_ps(a, b); }
    inline vec_ps vec_cvtepi32_ps(const vec_i32 a) { return _mm512_cvtepi32_ps(a); }
    inline vec_ps vec_loadu_ps(const float* a) { return _mm512_loadu_ps(a); }
    inline void vec_storeu_ps(float* a, const vec_ps b) { _mm512_storeu_ps(a, b); }

    inline vec_i16 vec_set1_epi16(const i16 a) { return _mm512_set1_epi16(a); }
    inline vec_i16 vec_setzero_epi16() { return _mm512_setzero_si512(); }
    inline vec_i16 vec_add_epi16(const vec_i16 a, const vec_i16 b) { return _mm512_add_epi16(a, b); }
    inline vec_i16 vec_sub_epi16(const vec_i16 a, const vec_i16 b) { return _mm512_sub_epi16(a, b); }
    inline vec_i16 vec_maddubs_epi16(const vec_i8 a, const vec_i8 b) { return _mm512_maddubs_epi16(a, b); }
    inline vec_i16 vec_mulhi_epi16(const vec_i16 a, const vec_i16 b) { return _mm512_mulhi_epi16(a, b); }
    inline vec_i16 vec_slli_epi16(const vec_i16 a, const i16 i) { return _mm512_slli_epi16(a, i); }
    inline vec_i16 vec_min_epi16(const vec_i16 a, const vec_i16 b) { return _mm512_min_epi16(a, b); }
    inline vec_i16 vec_max_epi16(const vec_i16 a, const vec_i16 b) { return _mm512_max_epi16(a, b); }
    inline vec_i16 vec_load_epi16(const vec_i16* a) { return _mm512_load_si512(a); }
    inline void vec_storeu_i16(vec_i16* a, const vec_i16 b) { _mm512_storeu_si512(a, b); }

    inline vec_i32 vec_set1_epi32(const i32 a) { return _mm512_set1_epi32(a); }
    inline vec_i32 vec_add_epi32(const vec_i32 a, const vec_i32 b) { return _mm512_add_epi32(a, b); }
    inline vec_i32 vec_madd_epi16(const vec_i16 a, const vec_i16 b) { return _mm512_madd_epi16(a, b); }

    inline uint16_t vec_nnz_mask(const vec_i32 vec) { return _mm512_cmpgt_epi32_mask(vec, _mm512_setzero_si512()); }

    inline float vec_hsum_ps(const vec_ps* v) {
        return _mm512_reduce_add_ps(v[0]);
    }

#elif defined(AVX256)

    using vec_i8 = __m256i;
    using vec_i16 = __m256i;
    using vec_i32 = __m256i;
    using vec_ps = __m256;

    inline vec_i8 vec_packus_epi16(const vec_i16 a, const vec_i16 b) { return _mm256_packus_epi16(a, b); }
    inline void vec_storeu_epi8(vec_i8* a, const vec_i8 b) { _mm256_storeu_si256(a, b); }

    inline vec_ps vec_set1_ps(const float a) { return _mm256_set1_ps(a); }
    inline vec_ps vec_fmadd_ps(const vec_ps a, const vec_ps b, const vec_ps c) { return _mm256_fmadd_ps(a, b, c); }
    inline vec_ps vec_min_ps(const vec_ps a, const vec_ps b) { return _mm256_min_ps(a, b); }
    inline vec_ps vec_max_ps(const vec_ps a, const vec_ps b) { return _mm256_max_ps(a, b); }
    inline vec_ps vec_mul_ps(const vec_ps a, const vec_ps b) { return _mm256_mul_ps(a, b); }
    inline vec_ps vec_cvtepi32_ps(const vec_i32 a) { return _mm256_cvtepi32_ps(a); }
    inline vec_ps vec_loadu_ps(const float* a) { return _mm256_loadu_ps(a); }
    inline void vec_storeu_ps(float* a, const vec_ps b) { _mm256_storeu_ps(a, b); }

    inline vec_i16 vec_set1_epi16(const i16 a) { return _mm256_set1_epi16(a); }
    inline vec_i16 vec_setzero_epi16() { return _mm256_setzero_si256(); }
    inline vec_i16 vec_maddubs_epi16(const vec_i8 a, const vec_i8 b) { return _mm256_maddubs_epi16(a, b); }
    inline vec_i16 vec_add_epi16(const vec_i16 a, const vec_i16 b) { return _mm256_add_epi16(a, b); }
    inline vec_i16 vec_sub_epi16(const vec_i16 a, const vec_i16 b) { return _mm256_sub_epi16(a, b); }
    inline vec_i16 vec_mulhi_epi16(const vec_i16 a, const vec_i16 b) { return _mm256_mulhi_epi16(a, b); }
    inline vec_i16 vec_slli_epi16(const vec_i16 a, const i16 i) { return _mm256_slli_epi16(a, i); }
    inline vec_i16 vec_min_epi16(const vec_i16 a, const vec_i16 b) { return _mm256_min_epi16(a, b); }
    inline vec_i16 vec_max_epi16(const vec_i16 a, const vec_i16 b) { return _mm256_max_epi16(a, b); }
    inline vec_i16 vec_load_epi16(const vec_i16* a) { return _mm256_load_si256(a); }
    inline void vec_storeu_i16(vec_i16* a, const vec_i16 b) { _mm256_storeu_si256(a, b); }

    inline vec_i32 vec_set1_epi32(const i32 a) { return _mm256_set1_epi32(a); }
    inline vec_i32 vec_add_epi32(const vec_i32 a, const vec_i32 b) { return _mm256_add_epi32(a, b); }
    inline vec_i32 vec_madd_epi16(const vec_i16 a, const vec_i16 b) { return _mm256_madd_epi16(a, b); }

    inline uint16_t vec_nnz_mask(const vec_i32 vec) { return _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(vec, _mm256_setzero_si256()))); }

    inline float vec_hsum_ps(const vec_ps* v) {
        const auto vec = _mm256_add_ps(v[0], v[1]);
        const auto sum_128 = _mm_add_ps(_mm256_castps256_ps128(vec), _mm256_extractf128_ps(vec, 1));
        const auto sum_64 = _mm_add_ps(sum_128, _mm_movehl_ps(sum_128, sum_128));
        const auto sum_32 = _mm_add_ss(sum_64, _mm_shuffle_ps(sum_64, sum_64, 1));

        return _mm_cvtss_f32(sum_32);
    }

#elif defined(AVX128)

    using vec_i8 = __m128i;
    using vec_i16 = __m128i;
    using vec_i32 = __m128i;
    using vec_ps = __m128;

    inline vec_i8 vec_packus_epi16(const vec_i16 a, const vec_i16 b) { return _mm_packus_epi16(a, b); }
    inline void vec_storeu_epi8(vec_i8* a, const vec_i8 b) { _mm_storeu_si128(a, b); }

    inline vec_ps vec_set1_ps(const float a) { return _mm_set1_ps(a); }
    inline vec_ps vec_fmadd_ps(const vec_ps a, const vec_ps b, const vec_ps c) { return _mm_fmadd_ps(a, b, c); }
    inline vec_ps vec_min_ps(const vec_ps a, const vec_ps b) { return _mm_min_ps(a, b); }
    inline vec_ps vec_max_ps(const vec_ps a, const vec_ps b) { return _mm_max_ps(a, b); }
    inline vec_ps vec_mul_ps(const vec_ps a, const vec_ps b) { return _mm_mul_ps(a, b); }
    inline vec_ps vec_cvtepi32_ps(const vec_i32 a) { return _mm_cvtepi32_ps(a); }
    inline vec_ps vec_loadu_ps(const float* a) { return _mm_loadu_ps(a); }
    inline void vec_storeu_ps(float* a, const vec_ps b) { _mm_storeu_ps(a, b); }

    inline vec_i16 vec_set1_epi16(const i16 a) { return _mm_set1_epi16(a); }
    inline vec_i16 vec_setzero_epi16() { return _mm_setzero_si128(); }
    inline vec_i16 vec_maddubs_epi16(const vec_i8 a, const vec_i8 b) { return _mm_maddubs_epi16(a, b); }
    inline vec_i16 vec_add_epi16(const vec_i16 a, const vec_i16 b) { return _mm_add_epi16(a, b); }
    inline vec_i16 vec_sub_epi16(const vec_i16 a, const vec_i16 b) { return _mm_sub_epi16(a, b); }
    inline vec_i16 vec_mulhi_epi16(const vec_i16 a, const vec_i16 b) { return _mm_mulhi_epi16(a, b); }
    inline vec_i16 vec_slli_epi16(const vec_i16 a, const i16 i) { return _mm_slli_epi16(a, i); }
    inline vec_i16 vec_min_epi16(const vec_i16 a, const vec_i16 b) { return _mm_min_epi16(a, b); }
    inline vec_i16 vec_max_epi16(const vec_i16 a, const vec_i16 b) { return _mm_max_epi16(a, b); }
    inline vec_i16 vec_load_epi16(const vec_i16* a) { return _mm_load_si128(a); }
    inline void vec_storeu_i16(vec_i16* a, const vec_i16 b) { _mm_storeu_si128(a, b); }

    inline vec_i32 vec_set1_epi32(const i32 a) { return _mm_set1_epi32(a); }
    inline vec_i32 vec_add_epi32(const vec_i32 a, const vec_i32 b) { return _mm_add_epi32(a, b); }
    inline vec_i32 vec_madd_epi16(const vec_i16 a, const vec_i16 b) { return _mm_madd_epi16(a, b); }

    inline uint16_t vec_nnz_mask(const vec_i32 vec) { return _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpgt_epi32(vec, _mm_setzero_si128()))); }

    inline float vec_hsum_ps(const vec_ps* v) {
        const auto vec = _mm_add_ps(_mm_add_ps(v[0], v[2]), _mm_add_ps(v[1], v[3]));
        const auto sum_64 = _mm_add_ps(vec, _mm_movehl_ps(vec, vec));
        const auto sum_32 = _mm_add_ss(sum_64, _mm_shuffle_ps(sum_64, sum_64, 1));

        return _mm_cvtss_f32(sum_32);
    }

#elif defined(ARM)

    using vec_i8 = int8x16_t;
    using vec_i16 = int16x8_t;
    using vec_i32 = int32x4_t;
    using vec_ps = float32x4_t;

    inline vec_i8 vec_packus_epi16(const vec_i16 a, const vec_i16 b) { return vcombine_u8(vqmovun_s16(a), vqmovun_s16(b)); }
    inline void vec_storeu_epi8(vec_i8* a, const vec_i8 b) { vst1q_s8(reinterpret_cast<i8*>(a), b); }

    inline vec_ps vec_set1_ps(const float a) { return vdupq_n_f32(a); }
    inline vec_ps vec_fmadd_ps(const vec_ps a, const vec_ps b, const vec_ps c) { return vfmaq_f32(c, a, b); }
    inline vec_ps vec_min_ps(const vec_ps a, const vec_ps b) { return vminq_f32(a, b); }
    inline vec_ps vec_max_ps(const vec_ps a, const vec_ps b) { return vmaxq_f32(a, b); }
    inline vec_ps vec_mul_ps(const vec_ps a, const vec_ps b) { return vmulq_f32(a, b); }
    inline vec_ps vec_cvtepi32_ps(const vec_i32 a) { return vcvtq_f32_s32(a); }
    inline vec_ps vec_loadu_ps(const float* a) { return vld1q_f32(a); }
    inline void vec_storeu_ps(float* a, const vec_ps b) { vst1q_f32(a, b); }

    inline vec_i16 vec_set1_epi16(const i16 a) { return vdupq_n_s16(a); }
    inline vec_i16 vec_setzero_epi16() { return vdupq_n_s16(0); }
    inline vec_i16 vec_add_epi16(const vec_i16 a, const vec_i16 b) { return vaddq_s16(a, b); }
    inline vec_i16 vec_sub_epi16(const vec_i16 a, const vec_i16 b) { return vsubq_s16(a, b); }

    inline vec_i16 vec_slli_epi16(const vec_i16 a, const i16 i) { return vshlq_s16(a, vdupq_n_s16(i)); }
    inline vec_i16 vec_min_epi16(const vec_i16 a, const vec_i16 b) { return vminq_s16(a, b); }
    inline vec_i16 vec_max_epi16(const vec_i16 a, const vec_i16 b) { return vmaxq_s16(a, b); }
    inline vec_i16 vec_load_epi16(const vec_i16* a) { return vld1q_s16(reinterpret_cast<const i16*>(a)); }
    inline void vec_storeu_i16(vec_i16* a, const vec_i16 b) { vst1q_s16(reinterpret_cast<i16*>(a), b); }

    inline vec_i32 vec_set1_epi32(const i32 a) { return vdupq_n_s32(a); }
    inline vec_i32 vec_add_epi32(const vec_i32 a, const vec_i32 b) { return vaddq_s32(a, b); }

    inline vec_i16 vec_maddubs_epi16(const vec_i8 a, const vec_i8 b) {
        const auto tl = vmulq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(a))), vmovl_s8(vget_low_s8(b)));
        const auto th = vmulq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(a))), vmovl_s8(vget_high_s8(b)));
        return vqaddq_s16(vuzp1q_s16(tl, th), vuzp2q_s16(tl, th));
    }

    inline vec_i16 vec_mulhi_epi16(const vec_i16 a, const vec_i16 b) {
        const auto lo = vmull_s16(vget_low_s16(a), vget_low_s16(b));
        const auto hi = vmull_s16(vget_high_s16(a), vget_high_s16(b));
        return vcombine_s16(vshrn_n_s32(lo, 16), vshrn_n_s32(hi, 16));
    }

    inline vec_i32 vec_madd_epi16(const vec_i16 a, const vec_i16 b) { 
        const auto lo = vmull_s16(vget_low_s16(a), vget_low_s16(b));
        const auto hi = vmull_high_s16(a, b);
        return vpaddq_s32(lo, hi);
    }

    inline uint16_t vec_nnz_mask(const vec_i32 vec) { 
        const auto mask = vcgtq_s32(vec, vec_setzero_epi16());
        const auto narrowed_mask = vmovn_u32(mask);
        const auto packed_mask = vget_lane_u64(vreinterpret_u64_u16(narrowed_mask), 0);
        const auto retVal = ((packed_mask & (1UL <<  0)) >>  0) |
                            ((packed_mask & (1UL << 16)) >> 15) |
                            ((packed_mask & (1UL << 32)) >> 30) |
                            ((packed_mask & (1UL << 48)) >> 45);
        return retVal;
    }

    inline float vec_hsum_ps(const vec_ps* v) {
        const auto sum02 = vaddq_f32(v[0], v[2]);
        const auto sum13 = vaddq_f32(v[1], v[3]);
        const auto reduced = vaddq_f32(sum02, sum13);
        return vaddvq_f32(reduced);
    }

#endif



    inline vec_128i vec128_set1_epi16(const i16 a) {
#ifdef ARM
        return vdupq_n_s16(a);
#else
        return _mm_set1_epi16(a);
#endif
    }

    inline vec_128i vec128_setzero_si128() {
        return vec128_set1_epi16(0);
    }

    inline void vec128_storeu_si128(vec_128i* a, const vec_128i b) {
#ifdef ARM
        vec_storeu_i16(a, b);
#else
        _mm_storeu_si128(a, b);
#endif
    }

    inline vec_128i vec128_add_epi16(const vec_128i a, const vec_128i b) {
#ifdef ARM
        return vaddq_s16(a, b);
#else
        return _mm_add_epi16(a, b);
#endif
    }


    inline vec_i32 vec_dpbusd_epi32(const vec_i32 sum, const vec_i8 vec0, const vec_i8 vec1) {
        const vec_i16 product16 = vec_maddubs_epi16(vec0, vec1);
        const vec_i32 product32 = vec_madd_epi16(product16, vec_set1_epi16(1));
        return vec_add_epi32(sum, product32);
    }

}
