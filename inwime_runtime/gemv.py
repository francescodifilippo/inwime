"""Contiguous bounded tiles, including cuts inside reduction dimensions.

Flat tiles can span many output rows. This avoids one I/O request per short row,
while carrying partial reductions in caller-owned FP64 output, never a full matrix.
"""
from array import array
from contextlib import closing
import ctypes
import time

from .core import open_catalog, Coordinator, Plan, CATEGORIES
from .pipeline import PipelineCoordinator
from .math_cpu import ScalarMath


def streamed_linear(control, math_backend, name, source, destination, offset=0):
    tensor = control.catalog.tensors[name]
    if len(tensor.shape) != 2:
        raise ValueError('GEMV requires a matrix')
    rows, columns = tensor.shape
    if (len(source) != columns or type(offset) is not int or offset < 0
            or len(destination) < offset+rows):
        raise ValueError('GEMV input/output shape mismatch')
    # Pin both exports for the whole operation, including GIL-releasing I/O.
    # Address ranges also detect distinct exporters sharing one allocation.
    source_view, output_view = memoryview(source), memoryview(destination)
    if any(v.readonly or not v.c_contiguous or v.format != 'd' or v.itemsize != 8
           for v in (source_view, output_view)):
        raise ValueError('GEMV requires writable contiguous FP64 arenas')
    source_address = ctypes.addressof(ctypes.c_char.from_buffer(source_view))
    output_address = ctypes.addressof(ctypes.c_char.from_buffer(output_view))+offset*8
    if (source_address < output_address+rows*8
            and output_address < source_address+columns*8):
        raise ValueError('GEMV input and output must not alias')
    requests = ((name, start, min(control.plan.tile_elements, tensor.elements-start))
                for start in range(0, tensor.elements, control.plan.tile_elements))
    with closing(control.stream(requests)) as stream:
        for start, count, lease in stream:
            began = time.monotonic_ns()
            lease.linear(math_backend, source, destination, columns, start, offset)
            control.gemv_ns += time.monotonic_ns()-began
            control.gemv_elements += count


class StreamedGEMV:
    """Weight acquisition only; input and output remain caller-owned and reported.

    The memory budget covers engine arenas and metadata. The caller must also
    budget 8*(input columns + output rows) bytes, plus its interpreter and OS costs.
    """
    def __init__(self, directory, *, tile_elements=262144, read_elements=262144,
                 io_mode='sync', math_backend=None, budget=None, backend=None,
                 integrity='full', source='fixture'):
        if (io_mode not in ('sync', 'pipeline') or type(tile_elements) is not int
                or type(read_elements) is not int or not 0 < read_elements <= tile_elements):
            raise ValueError('invalid GEMV tile/read/mode')
        catalog = open_catalog(directory, source=source)
        slots = 2 if io_mode == 'pipeline' else 1
        arrays = {}
        for slot in range(slots):
            suffix = str(slot) if slot else ''
            arrays['weights'+suffix] = ('weights', tile_elements, 'd')
            arrays['staging'+suffix] = ('staging', 2*(tile_elements if slots > 1 else read_elements), 'B')
        required = dict.fromkeys(CATEGORIES, 0)
        for category, count, code in arrays.values():
            required[category] += count*array(code).itemsize
        required['metadata'] = catalog.bootstrap_bytes+65536
        required['requests'] = 24576 if slots > 1 else 0
        plan = Plan(arrays, required, 1, tile_elements, read_elements,
                    slots if slots > 1 else 0, 1 if slots > 1 else 0)
        self.math = ScalarMath() if math_backend is None else math_backend
        coordinator = Coordinator if slots == 1 else PipelineCoordinator
        self.control = coordinator(catalog, plan, budget, backend, integrity=integrity,
                                   math_backend=self.math)

    def run(self, tensor, source, destination, *, cancel_event=None):
        self.control.check_thread()
        if self.control.closed or self.control.active:
            raise RuntimeError('closed or busy GEMV')
        try:
            self.control.cancel_event = cancel_event
            self.control.check_cancelled()
            streamed_linear(self.control, self.math, tensor, source, destination)
            self.control.check_cancelled()
        finally:
            self.control.cancel_event = None

    def stats(self):
        return self.control.stats() | {'math_backend': self.math.name}

    def close(self):
        self.control.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
