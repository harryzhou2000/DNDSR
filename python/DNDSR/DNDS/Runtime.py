"""Native row Arrays. NumPy results are snapshots; mutation needs a write lease.

MPI plan construction, binding and begin are collective on the plan's MPIInfo.
An operation owns its leases until completion. Observe errors with explicit wait().
Eager callers must join cross-stream conflicts and all uses before final release;
scheduled reservations do not prove ordering. Finish asynchronous producers before
synchronous indexing/checkpointing. Array reset does not retarget retained views
or MPI bindings. Match collective order/tags on all ranks, including empty ranks,
and drive MPI progress on a thread allowed by MPI's initialized thread level.
See docs/architecture/RuntimeCallerContract.md for the complete caller contract.
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
    """A live debugging view retaining the shared ArrayStorage, independently
    of its parent handle: values, row metadata, device offsets and controller.

    Basic integer/slice indexing only. Reading CUDA values synchronizes and copies
    the selected values to host. numpy() always returns an independent snapshot.
    Holding a view does not hold a lease; each access acquires its own lease.
    Complete pending asynchronous writes before indexing. Resetting the parent
    leaves this view attached to the old storage and old row metadata.
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
        return selected._scalar() if not selected._axes else selected

    def _scalar(self):
        if self._base.element_type is not None:
            return self._base._read_object(self._rows[0], self._columns[0])
        return self.numpy()[()]

    def _assign(self, value):
        if self._base.element_type is not None:
            if isinstance(value, ElementRef):
                value = value.snapshot()
            if not self._axes:
                self._base._write_object(self._rows[0], self._columns[0], value)
                return
            objects = np.empty(self.shape, dtype=object)
            objects[...] = value
            objects = objects.reshape(len(self._rows), len(self._columns))
            for i, row in enumerate(self._rows):
                for j, column in enumerate(self._columns):
                    item = objects[i, j]
                    self._base._write_object(row, column, item.snapshot() if isinstance(item, ElementRef) else item)
            return
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
    rows, row_scalar = _select_range(range(array.layout.row_count), row_key)
    if array.layout.is_variable:
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
    return selected._scalar() if not selected._axes else selected


def _array_setitem(array, key, value):
    _array_selection(array, key)._assign(value)


class ElementRef:
    """Tracked live POD element, retaining ArrayStorage and its POD class codec
    through Array reset/destruction, without retaining a permanent access lease.

    Attribute writes and method calls edit a class-value snapshot under one
    write lease and copy it back, synchronizing CUDA. Attribute reads return
    snapshots. Use edit(callback) for nested mutable attributes or several edits.
    """

    def __init__(self, array, row, column):
        selected = _array_selection(array, (row, column))
        if selected._axes or array.element_type is None:
            raise TypeError("element() requires a registered POD class and two scalar indices")
        object.__setattr__(self, "_array", array.alias())
        object.__setattr__(self, "_row", selected._rows[0])
        object.__setattr__(self, "_column", selected._columns[0])

    def snapshot(self):
        return self._array._read_object(self._row, self._column)

    def assign(self, value):
        self._array._write_object(self._row, self._column, value)

    def edit(self, callback):
        return self._array._edit_object(self._row, self._column, callback)

    def __getattr__(self, name):
        value = getattr(self.snapshot(), name)
        if callable(value):
            return lambda *args, **kwargs: self.edit(lambda item: getattr(item, name)(*args, **kwargs))
        return value

    def __setattr__(self, name, value):
        self.edit(lambda item: setattr(item, name, value))

    def __repr__(self):
        return f"ElementRef({self.snapshot()!r})"


# Keep the pybind type itself: aliases, placement copies and deserialization all
# return this same class. Views retain storage, not the replaceable Python handle.
Array.__getitem__ = _array_getitem
Array.__setitem__ = _array_setitem
Array.__len__ = lambda self: self.layout.row_count
Array.view = lambda self: _array_selection(self, (slice(None), slice(None)))
Array.shape = property(lambda self: (self.layout.row_count, None if self.layout.is_variable else self.layout.width))
Array.element = lambda self, row, column=0: ElementRef(self, row, column)


class ArrayPair:
    """Separate owned/ghost allocations with optional combined read indexing."""

    def __init__(self, owned, ghost):
        if owned.dtype != ghost.dtype or owned.element_type is not ghost.element_type:
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
            self._n_owned = pair.owned.layout.row_count
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
