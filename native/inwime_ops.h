/* Leaf operators of the InWiMe Percorso additivo (#41, ADR 0001 and 0002). Stable C ABI.
 *
 * Pure functions over caller-owned FP64 arenas: no allocation, no I/O, no threads, no global
 * state. Each one reproduces, element by element and in the same order of operations, a step of
 * the frozen Python Qwen2 forward (inwime_runtime/llama.py), so the results keep its bits on the
 * certified configuration. Build without fast-math, without FP contraction and without builtin
 * math (no sincos fusion, no constant folding): see setup.py and the Makefile.
 *
 * Arenas: element offsets are applied by the caller; `*_stride` is the distance, in elements,
 * between the rows of two consecutive positions (0 repeats one row, as for a bias).
 * Status: INWIME_OPS_OK, _NONFINITE (an overflow the frozen path rejects), _CANCELLED (the flag
 * was nonzero), _INVALID (inconsistent dimensions; shapes are validated by the caller first).
 *
 * Cancellation frequency: inwime_ops_linear reads *cancel before each row segment of the tile
 * (at most positions*columns multiply-adds between two reads); inwime_ops_attention before each
 * (position, head) pair (at most 3*(context)*dim multiply-adds). The other operators are linear
 * in the block and are not interrupted.
 */
#ifndef INWIME_OPS_H
#define INWIME_OPS_H

#include <stddef.h>

#define INWIME_OPS_ABI_VERSION 1

enum { INWIME_OPS_OK = 0, INWIME_OPS_NONFINITE = 1, INWIME_OPS_CANCELLED = 2, INWIME_OPS_INVALID = 3 };

int inwime_ops_abi_version(void);

/* y[p*ys + row] for the `count` weights w[0..count) of the flat matrix starting at element `start`. */
int inwime_ops_linear(const double *w, size_t count, const double *x, size_t x_stride,
                      double *y, size_t y_stride, size_t positions, size_t columns, size_t start,
                      const volatile int *cancel);

int inwime_ops_rms_denominators(const double *x, size_t x_stride, size_t positions, size_t width,
                                double eps, double *out);

/* y[p*ys + start + i] = x[p*xs + start + i] / d[p] * w[i] for i < count. */
int inwime_ops_rms_scale(const double *x, size_t x_stride, const double *denominators,
                         const double *w, size_t count, size_t start, double *y, size_t y_stride,
                         size_t positions);

int inwime_ops_add(double *y, size_t y_stride, const double *x, size_t x_stride, size_t positions,
                   size_t count);

int inwime_ops_rope(double *v, size_t stride, size_t positions, size_t first, size_t heads,
                    size_t dim, double theta);

/* keys/values: row t of the layer at keys + t*kv_stride (resp. values). */
int inwime_ops_attention(const double *q, size_t q_stride, size_t positions, size_t first,
                         const double *keys, const double *values, size_t kv_stride, size_t heads,
                         size_t kv_heads, size_t dim, double *out, size_t out_stride,
                         const volatile int *cancel);

int inwime_ops_silu_mul(double *gate, const double *up, size_t count);

#endif
