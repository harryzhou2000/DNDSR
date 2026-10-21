"""Native row Arrays. NumPy results are snapshots; mutation needs a write lease.

MPI plan construction, binding and begin are collective on the plan's MPIInfo.
An operation owns its leases until wait() or destruction. Prefer explicit wait().
"""

from contextlib import ExitStack
import operator
import numpy as np
from ._ext.dnds_pybind11.Runtime import *  # noqa: F401,F403


def _indices(key, ndim):
    key = key if isinstance(key, tuple) else (key,)
    if sum(part is Ellipsis for part in key) > 1:
        raise IndexError("only one ellipsis is supported")
    if any(part is Ellipsis for part in key):
        position = next(i for i, part in enumerate(key) if part is Ellipsis)
        key = key[:position] + (slice(None),) * (ndim - len(key) + 1) + key[position + 1:]
    if len(key) > ndim:
        raise IndexError("too many Array indices")
    return key + (slice(None),) * (ndim - len(key))


def _select_range(values, key):
    if isinstance(key, slice):
        return values[key], False
    try:
        index = operator.index(key)
    except TypeError as error:
        raise IndexError("Array debugging views support integers and slices") from error
    index = values[index]  # range implements negative indexing and bounds checking
    return range(index, index + 1), True


class ArrayView:
    """A live debugging view retaining one allocation and its access controller.

    Basic integer/slice indexing only. Reading CUDA values synchronizes and copies
    the selected values to host. numpy() always returns an independent snapshot.
    This is not an MPI payload: its columns may no longer describe complete rows.
    """

    def __init__(self, base, rows, columns, axes=(0, 1)):
        self._base = base.alias()
        self._rows, self._columns, self._axes = rows, columns, axes

    @property
    def shape(self):
        dimensions = (len(self._rows), len(self._columns))
        return tuple(dimensions[axis] for axis in self._axes)

    @property
    def dtype(self):
        return self._base.dtype

    @property
    def device(self):
        return self._base.device

    def _select(self, key):
        ranges = [self._rows, self._columns]
        axes = []
        for axis, part in zip(self._axes, _indices(key, len(self._axes))):
            ranges[axis], scalar = _select_range(ranges[axis], part)
            if not scalar:
                axes.append(axis)
        return ArrayView(self._base, *ranges, tuple(axes))

    def numpy(self):
        return self._base._read_elements(self._rows, self._columns).reshape(self.shape)

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("a tracked Array view cannot export an unleased NumPy alias")
        result = self.numpy()
        return result.astype(dtype, copy=False) if dtype is not None else result

    def __getitem__(self, key):
        selected = self._select(key)
        return selected.numpy()[()] if not selected._axes else selected

    def _assign(self, value):
        values = np.empty(self.shape, dtype=self.dtype)
        values[...] = value  # Validate/broadcast before taking a write lease.
        self._base._write_elements(self._rows, self._columns,
                                   values.reshape(len(self._rows), len(self._columns)))

    def __setitem__(self, key, value):
        self._select(key)._assign(value)

    def item(self):
        return self.numpy().item()

    def tolist(self):
        return self.numpy().tolist()

    def __len__(self):
        if not self.shape:
            raise TypeError("len() of scalar view")
        return self.shape[0]

    def __repr__(self):
        return f"ArrayView({self.numpy()!r}, device={self.device.space})"


def _array_selection(array, key):
    row_key, column_key = _indices(key, 2)
    rows, row_scalar = _select_range(range(array.layout.rows), row_key)
    if array.layout.is_ragged:
        widths = {array.layout.row_size(row) for row in rows}
        if len(widths) > 1:
            raise IndexError("select one row or equal-width rows from a ragged Array")
        width = next(iter(widths), 0)
    else:
        width = array.layout.width
    columns, column_scalar = _select_range(range(width), column_key)
    axes = tuple(axis for axis, scalar in enumerate((row_scalar, column_scalar)) if not scalar)
    return ArrayView(array, rows, columns, axes)


def _array_getitem(array, key):
    selected = _array_selection(array, key)
    return selected.numpy()[()] if not selected._axes else selected


def _array_setitem(array, key, value):
    _array_selection(array, key)._assign(value)


# Keep the pybind type itself: aliases, placement copies and deserialization all
# return this same class. Views retain storage, not the replaceable Python handle.
Array.__getitem__ = _array_getitem
Array.__setitem__ = _array_setitem
Array.__len__ = lambda self: self.layout.rows
Array.view = lambda self: _array_selection(self, (slice(None), slice(None)))
Array.shape = property(lambda self: (self.layout.rows, None if self.layout.is_ragged else self.layout.width))


class ArrayPair:
    """Separate owned/ghost allocations with optional combined read indexing."""

    def __init__(self, owned, ghost):
        if owned.dtype != ghost.dtype:
            raise TypeError("ArrayPair dtype mismatch")
        if owned.identity == ghost.identity:
            raise ValueError("owned and ghost must use separate allocations")
        self.owned = owned.alias()
        self.ghost = ghost.alias()

    def read(self, stream=0):
        return _PairRead(self, stream)


class _PairRead:
    def __init__(self, pair, stream):
        self._stack = ExitStack()
        try:
            self._owned = self._stack.enter_context(pair.owned.read(stream))
            self._ghost = self._stack.enter_context(pair.ghost.read(stream))
            self._n_owned = pair.owned.layout.rows
        except BaseException:
            self._stack.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self._stack.__exit__(*args)

    def __getitem__(self, index):
        row, column = index
        if row < self._n_owned:
            return self._owned[row, column]
        return self._ghost[row - self._n_owned, column]

    def wait(self):
        self._stack.close()

    def complete(self, event):
        self._owned.complete(event)
        self._ghost.complete(event)
