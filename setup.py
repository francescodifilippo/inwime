"""Optional native CPU kernels, loaded through ctypes, not imported as modules.

- `inwime._cpu_kernel`: the pinned increment-10 source `src/cpu_kernel.c`, compiled without
  changes. The numerically relevant flags of the historical `make cpu-kernel` build are added
  (-O3, no fast-math, no FP contraction), but setuptools prepends Python's own CFLAGS and linker
  settings, so the two builds are not byte-identical. Bit-identity of their results is a test,
  not an assumption: see the packaging test, run in CI against `make cpu-kernel`.
- `inwime._ops_kernel`: the operators of the Percorso additivo (#41), `src/inwime_ops.c`, with
  the same flags plus -fno-builtin (no sincos fusion or constant folding of libm calls). Each
  operator is checked bit for bit against its Python reference when the Motore loads
  (`inwime/gate.py`); one that differs is not used.

`optional=True` lets the installation succeed without a compiler, leaving the Python fallback.
"""
from setuptools import Extension, setup

FLAGS = ['-O3', '-std=c11', '-fno-fast-math', '-ffp-contract=off']

setup(ext_modules=[
    Extension('inwime._cpu_kernel', sources=['src/cpu_kernel.c'], extra_compile_args=FLAGS,
              optional=True),
    Extension('inwime._ops_kernel', sources=['src/inwime_ops.c'], depends=['src/inwime_ops.h'],
              include_dirs=['src'], extra_compile_args=FLAGS+['-fno-builtin'], libraries=['m'],
              optional=True)])
