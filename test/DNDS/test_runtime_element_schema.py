"""Structured NumPy schemas use exact word layouts and stable identities."""

import numpy as np
import pytest
from mpi4py import MPI
from DNDSR import DNDS
from DNDSR.DNDS import Runtime as R


def local_route(rows=2):
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    rank = MPI.COMM_WORLD.rank
    return R.RowTransferPlan(mpi, rows, rows, [R.RowRequest(rank, row, row) for row in range(rows)])


@pytest.mark.parametrize("mixed", [False, True])
def test_structured_schema_rejects_same_size_different_words_collectively(mixed):
    layout = R.RowLayout.dense(2, 1)
    correct = np.dtype([("x", "i8"), ("y", "f8")])
    wrong = np.dtype([("other", "f8"), ("z", "i8")])
    prototype = R.Array(layout, correct)
    dtype = correct if mixed and MPI.COMM_WORLD.rank % 2 else wrong
    source, target = R.Array(layout, dtype), R.Array(layout, dtype)
    route = local_route()
    schema = prototype.element_schema()
    plan = R.PayloadPlan(route, layout, layout, schema)
    with pytest.raises(RuntimeError):
        plan.bind(source, target)
    with pytest.raises(RuntimeError, match="schema"):
        R.DynamicRowTransfer(source, route, schema)
    assert source.valid and target.valid
    # A failed collective check must permit a correctly typed retry.
    retry = R.Array(layout, correct)
    plan.bind(prototype, retry).begin().wait()


@pytest.mark.parametrize("dtype", [np.dtype("float16"), np.dtype("longdouble"),
                                  np.dtype("complex64"), np.dtype("complex128"),
                                  np.dtype("U3")])
@pytest.mark.parametrize("nested", [False, True])
def test_unsupported_word_dtypes_are_rejected_before_transfer_or_io(dtype, nested):
    # NumPy may alias longdouble to float64 on some platforms.
    if dtype == np.dtype("float64"):
        pytest.skip("this platform uses a supported binary64 long double")
    if nested:
        dtype = np.dtype([("header", "u1"), ("unsupported", dtype)])
    with pytest.raises(RuntimeError, match="codec"):
        R.Array(R.RowLayout.dense(1, 1), dtype)


def test_packed_record_keeps_unaligned_scalar_word_offsets():
    dtype = np.dtype([("tag", "u1"), ("value", "i4")])
    assert dtype.itemsize == 5 and dtype.alignment == 1
    source = R.Array(R.RowLayout.dense(2, 1), dtype)
    values = np.zeros((2, 1), dtype=dtype)
    values["tag"][:, 0] = [7, 9]
    values["value"][:, 0] = [0x12345678, -271]
    source.assign(values)
    schema = source.element_schema()
    assert schema.word_fields == [("tag", 0, "uint8", 1), ("value", 1, "int32", 1)]
    target = R.Array(source.layout, dtype)
    route = local_route()
    R.PayloadPlan(route, source.layout, target.layout, schema).bind(source, target).begin().wait()
    assert np.array_equal(target.numpy(), values)


@pytest.mark.parametrize("dtype", [
    np.dtype([("a.b", "i4"), ("a", [("b", "i4")])]),
    np.dtype([("a[0]", "i4"), ("a", "i4", (1,))]),
    np.dtype([("a\\b", "i4"), ("a.b", "i4")]),
])
def test_flattened_names_distinguish_literal_delimiters_from_paths(dtype):
    array = R.Array(R.RowLayout.dense(1, 1), dtype)
    words = array.element_schema().word_fields
    assert len(words) == 2
    assert len({word[0] for word in words}) == 2
    assert [word[1] for word in words] == [0, 4]
    assert all(word[2:] == ("int32", 1) for word in words)


def test_subarray_words_use_base_offsets_and_stable_little_endian_identity():
    dtype = np.dtype([("id", "i8"), ("lanes", "i2", (2, 3)), ("tail", "V3")])
    schema = R.Array(R.RowLayout.dense(1, 1), dtype).element_schema()
    assert schema.identity == f"numpy.pod.v1:{dtype.newbyteorder('<').descr}:{dtype.itemsize}"
    assert schema.can_swap_words
    words = schema.word_fields
    assert words[0] == ("id", 0, "int64", 1)
    assert words[1:7] == [(f"lanes[{i}]", 8 + i * 2, "int16", 1) for i in range(6)]
    assert words[7] == ("tail", 20, "byte", 3)


def test_opaque_void_dtype_declares_bytes_and_preserves_simple_native_field_names():
    opaque = R.Array(R.RowLayout.dense(1, 1), np.dtype("V7")).element_schema()
    assert opaque.word_fields == [("bytes", 0, "byte", 7)]
    assert opaque.can_swap_words
    plain = R.Array(R.RowLayout.dense(1, 1), np.dtype([("id", "i8"), ("weight", "f8")]))
    assert [word[0] for word in plain.element_schema().word_fields] == ["id", "weight"]


def test_non_native_scalar_and_nested_byte_orders_still_reject_input():
    foreign = ">" if np.little_endian else "<"
    for dtype in [np.dtype(foreign + "i4"), np.dtype([("word", foreign + "f8")])]:
        with pytest.raises(RuntimeError, match="native"):
            R.Array(R.RowLayout.dense(1, 1), dtype)
