"""Optional native CPU kernel, compiled from the pinned increment-10 source without changes.

The library is loaded through ctypes, not imported as a module. The numerically relevant
flags of the historical `make cpu-kernel` build are added (-O3, no fast-math, no FP
contraction), but setuptools prepends Python's own CFLAGS and linker settings, so the two
builds are not byte-identical. Bit-identity of their results is a test, not an assumption:
see the packaging test, run in CI against `make cpu-kernel`. `optional=True` lets the
installation succeed without a compiler, leaving the Python FP64 fallback.
"""
from setuptools import Extension, setup

setup(ext_modules=[Extension(
    'inwime._cpu_kernel', sources=['src/cpu_kernel.c'],
    extra_compile_args=['-O3', '-std=c11', '-fno-fast-math', '-ffp-contract=off'],
    optional=True)])
