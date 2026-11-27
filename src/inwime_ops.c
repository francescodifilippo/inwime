/* Leaf operators of the Percorso additivo; contract and status codes in inwime_ops.h.
 * Separate from the pinned src/cpu_kernel.c, which stays unchanged.
 * Each loop mirrors the Python reference of inwime/ops.py operation for operation.
 */
#include "inwime_ops.h"

#include <float.h>
#include <math.h>

_Static_assert(sizeof(double) == 8 && DBL_MANT_DIG == 53 && FLT_RADIX == 2, "IEEE binary64 required");

int inwime_ops_abi_version(void) { return INWIME_OPS_ABI_VERSION; }

static int cancelled(const volatile int *cancel) { return cancel && *cancel; }

int inwime_ops_linear(const double *w, size_t count, const double *x, size_t x_stride,
                      double *y, size_t y_stride, size_t positions, size_t columns, size_t start,
                      const volatile int *cancel) {
    if (!columns) return INWIME_OPS_INVALID;
    size_t row = start / columns, column = start % columns, i = 0;
    while (i < count) {
        if (cancelled(cancel)) return INWIME_OPS_CANCELLED;
        size_t number = columns-column;
        if (number > count-i) number = count-i;
        for (size_t p = 0; p < positions; ++p) {
            const double *xp = x + p*x_stride + column;
            double *target = y + p*y_stride + row;
            double total = column ? *target : 0.;
            for (size_t j = 0; j < number; ++j) total += w[i+j]*xp[j];
            *target = total;
            if (column+number == columns && !isfinite(total)) return INWIME_OPS_NONFINITE;
        }
        i += number;
        column += number;
        if (column == columns) {
            ++row;
            column = 0;
        }
    }
    return INWIME_OPS_OK;
}

int inwime_ops_rms_denominators(const double *x, size_t x_stride, size_t positions, size_t width,
                                double eps, double *out) {
    if (!width) return INWIME_OPS_INVALID;
    for (size_t p = 0; p < positions; ++p) {
        const double *xp = x + p*x_stride;
        double squared = 0.;
        for (size_t i = 0; i < width; ++i) squared += xp[i]*xp[i];
        double denominator = sqrt(squared/(double)width + eps);
        if (!isfinite(denominator)) return INWIME_OPS_NONFINITE;
        out[p] = denominator;
    }
    return INWIME_OPS_OK;
}

int inwime_ops_rms_scale(const double *x, size_t x_stride, const double *denominators,
                         const double *w, size_t count, size_t start, double *y, size_t y_stride,
                         size_t positions) {
    for (size_t p = 0; p < positions; ++p) {
        const double *xp = x + p*x_stride + start;
        double *yp = y + p*y_stride + start, d = denominators[p];
        for (size_t i = 0; i < count; ++i) yp[i] = xp[i]/d*w[i];
    }
    return INWIME_OPS_OK;
}

int inwime_ops_add(double *y, size_t y_stride, const double *x, size_t x_stride, size_t positions,
                   size_t count) {
    for (size_t p = 0; p < positions; ++p) {
        double *yp = y + p*y_stride;
        const double *xp = x + p*x_stride;
        for (size_t i = 0; i < count; ++i) yp[i] = yp[i]+xp[i];
    }
    return INWIME_OPS_OK;
}

int inwime_ops_rope(double *v, size_t stride, size_t positions, size_t first, size_t heads,
                    size_t dim, double theta) {
    if (!dim || dim % 2) return INWIME_OPS_INVALID;
    size_t half = dim/2;
    for (size_t p = 0; p < positions; ++p) {
        double position = (double)(first+p);
        for (size_t head = 0; head < heads; ++head) {
            for (size_t j = 0; j < half; ++j) {
                double angle = position / pow(theta, (double)(2*j)/(double)dim);
                double cosine = cos(angle), sine = sin(angle);
                double *a = v + p*stride + head*dim + j, *b = a + half;
                double xa = *a, yb = *b;
                *a = xa*cosine-yb*sine;
                *b = yb*cosine+xa*sine;
            }
        }
    }
    return INWIME_OPS_OK;
}

static double score(const double *q, const double *k, size_t dim) {
    double total = 0.;
    for (size_t d = 0; d < dim; ++d) total += q[d]*k[d];
    return total/sqrt((double)dim);
}

int inwime_ops_attention(const double *q, size_t q_stride, size_t positions, size_t first,
                         const double *keys, const double *values, size_t kv_stride, size_t heads,
                         size_t kv_heads, size_t dim, double *out, size_t out_stride,
                         const volatile int *cancel) {
    if (!dim || !kv_heads || heads % kv_heads) return INWIME_OPS_INVALID;
    size_t group = heads/kv_heads;
    for (size_t p = 0; p < positions; ++p) {
        size_t position = first+p;
        for (size_t head = 0; head < heads; ++head) {
            if (cancelled(cancel)) return INWIME_OPS_CANCELLED;
            size_t kh = head/group;
            const double *qh = q + p*q_stride + head*dim;
            double *oh = out + p*out_stride + head*dim;
            /* Python's max(): the first item, replaced only by a strictly greater one. */
            double maximum = score(qh, keys + kh*dim, dim);
            for (size_t t = 1; t <= position; ++t) {
                double s = score(qh, keys + t*kv_stride + kh*dim, dim);
                if (s > maximum) maximum = s;
            }
            double denominator = 0.;
            for (size_t d = 0; d < dim; ++d) oh[d] = 0.;
            for (size_t t = 0; t <= position; ++t) {
                double probability = exp(score(qh, keys + t*kv_stride + kh*dim, dim)-maximum);
                denominator += probability;
                const double *vt = values + t*kv_stride + kh*dim;
                for (size_t d = 0; d < dim; ++d) oh[d] = oh[d] + probability*vt[d];
            }
            for (size_t d = 0; d < dim; ++d) oh[d] = oh[d]/denominator;
        }
    }
    return INWIME_OPS_OK;
}

int inwime_ops_silu_mul(double *gate, const double *up, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        double z = gate[i], silu;
        if (z >= 0) {
            silu = z/(1.+exp(-z));
        } else {
            double e = exp(z);
            silu = z*e/(1.+e);
        }
        gate[i] = silu*up[i];
    }
    return INWIME_OPS_OK;
}
