#pragma once

/* Portable SIMD primitives for CATALYST's cursor table (paper Sec. 4.3).
 *
 * We use GCC/clang vector extensions rather than raw intrinsics so one source
 * compiles to AVX2, AVX-512, NEON or RVV. The width is chosen at compile time
 * from -march, so the dev box (i9-9900K, AVX2 -> 4 lanes of uint64) and the
 * eval cluster (Xeon Gold 6242R, AVX-512 -> 8 lanes) share this file.
 *
 * Note the lane count differs from the paper's figure, which assumes 32-bit
 * keys for 16-way parallelism. DEX's Key is uint64_t (Common.h), so a 512-bit
 * register holds 8 bands, not 16. KeyT is a template parameter throughout, so
 * a truncated-prefix variant is a configuration change and not a rewrite.
 */

#include <cstdint>
#include <cstring>
#include <type_traits>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace catalyst {
namespace simd {

// Widest vector the target actually supports, in bytes.
#if defined(__AVX512F__)
static constexpr int kNativeBytes = 64;
#elif defined(__AVX2__)
static constexpr int kNativeBytes = 32;
#else
static constexpr int kNativeBytes = 16;
#endif

// Vector of Lanes x T. GCC vector extensions give us portable elementwise
// compares; the result is a same-shape vector with all-ones in matching lanes.
template <typename T, int Bytes> struct GccVec {
  typedef T type __attribute__((vector_size(Bytes)));
  static constexpr int kLanes = Bytes / sizeof(T);
};

template <typename VecT> inline VecT load(const void *p) {
  VecT v;
  std::memcpy(&v, p, sizeof(VecT));
  return v;
}

/* Collapse a comparison result into one bit per lane.
 *
 * GCC 9 (this box) lacks __builtin_reduce_or, and movemask is not portable, so
 * this is a small ISA dispatch: x86 uses the native instruction, everything
 * else falls back to a scalar loop that -O3 usually recognises anyway. Lanes
 * are all-ones or all-zero, so testing the sign bit is sufficient.
 */
template <typename MaskT>
inline uint32_t lane_bits(MaskT m, int lanes) {
  // Subscripting a vector yields a reference, hence the decay.
  using Lane = typename std::decay<decltype(m[0])>::type;
  constexpr int kN = (int)(sizeof(MaskT) / sizeof(Lane));

  /* These must be `if constexpr`, not `if`. An AVX-512 target also defines
   * __AVX2__, so both blocks are compiled; with a plain `if` the narrower
   * branch is still type-checked and casting a 64-byte vector to __m256d is a
   * hard error even though that branch can never be taken. */
#if defined(__AVX512DQ__)
  // movepi64_mask is AVX512DQ, not plain AVX512F; an F-only target falls
  // through to the generic path below rather than failing to compile.
  if constexpr (sizeof(MaskT) == 64 && sizeof(Lane) == 8) {
    return (uint32_t)_mm512_movepi64_mask((__m512i)m);
  } else
#endif
#if defined(__AVX2__)
  if constexpr (sizeof(MaskT) == 32 && sizeof(Lane) == 8) {
    return (uint32_t)_mm256_movemask_pd((__m256d)m);
  } else
#endif
  {
    // Generic: read the lanes back and test each sign bit. Lanes are all-ones
    // or all-zero, so a nonzero test is sufficient. -O3 often recognises this.
    uint32_t bits = 0;
    alignas(64) Lane tmp[kN];
    std::memcpy(tmp, &m, sizeof(MaskT));
    for (int i = 0; i < lanes; ++i) {
      bits |= (tmp[i] != 0) ? (1u << i) : 0u;
    }
    return bits;
  }
}

// Iterate set bits low-to-high: while (bits) { int i = next_lane(bits); ... }
inline int next_lane(uint32_t &bits) {
  int i = __builtin_ctz(bits);
  bits &= (bits - 1);
  return i;
}

} // namespace simd
} // namespace catalyst
