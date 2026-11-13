/* Original BF16 -> exact FP64, ordered GEMV, no allocator, no I/O, no GPU.
 * Build without fast-math or contraction: reduction order is part of this backend.
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>

_Static_assert(sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24,
               "IEEE binary32 required");
_Static_assert(sizeof(double) == 8 && DBL_MANT_DIG == 53, "IEEE binary64 required");

int inwime_expand(const unsigned char *raw, size_t count, double *out) {
    for (size_t i = 0; i < count; ++i) {
        uint32_t word = (uint32_t)raw[2*i] | (uint32_t)raw[2*i+1] << 8;
        if ((word & 0x7f80u) == 0x7f80u) return 1;
        uint32_t bits = word << 16;
        float value;
        memcpy(&value, &bits, sizeof(value));
        out[i] = (double)value;
    }
    return 0;
}

int inwime_linear(const double *weights, size_t count, const double *x,
                  double *y, size_t columns, size_t start) {
    size_t row = start / columns, column = start % columns, i = 0;
    while (i < count) {
        double sum = column ? y[row] : 0.;
        size_t number = columns-column;
        if (number > count-i) number = count-i;
        for (size_t j = 0; j < number; ++j) sum += weights[i+j]*x[column+j];
        y[row] = sum;
        i += number;
        column += number;
        if (column == columns) {
            if (!isfinite(sum)) return 1;
            ++row;
            column = 0;
        }
    }
    return 0;
}

/* SplitMix64 finalizer with explicitly wrapping unsigned arithmetic. */
void inwime_fill(unsigned char *raw, size_t count, uint64_t start,
                 uint64_t seed, uint32_t exponent, int norm) {
    for (size_t i = 0; i < count; ++i) {
        uint64_t z = (start+i+seed) + UINT64_C(0x9e3779b97f4a7c15);
        z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
        z ^= z >> 31;
        uint32_t word = norm ? 0x3f80u :
            (uint32_t)((z >> 48) & 0x8000u) | (exponent << 7) | (uint32_t)(z & 127);
        raw[2*i] = (unsigned char)word;
        raw[2*i+1] = (unsigned char)(word >> 8);
    }
}
