"""Published row shapes, incremental construction and explicit exclusive edits."""

import gc
import numpy as np
import pytest
from DNDSR.DNDS import Runtime as R


def test_metadata_snapshots_and_padded_debug_indexing():
    csr = R.RowLayout.ragged([0, 5, 7])
    assert csr.offsets == [0, 0, 5, 12]
    detached_offsets = csr.offsets
    detached_offsets[2] = 19
    assert csr.offsets == [0, 0, 5, 12]
    padded = R.RowLayout.padded([0, 5, 7], 8)
    assert padded.sizes == [0, 5, 7] and padded.offsets == []
    assert padded.kind == R.RowLayoutKind.Padded
    a = R.Array(padded)
    assert a.shape == (3, None)
    a[2] = np.arange(7)
    assert a[2, -1] == 6
    with pytest.raises(RuntimeError):
        a.numpy()
    with pytest.raises((IndexError, RuntimeError)):
        a[0, 0]


@pytest.mark.parametrize("kind", [R.RowLayoutKind.CSR, R.RowLayoutKind.Padded])
def test_incremental_rows_commit_once_and_keep_draft_snapshots_independent(kind):
    builder = R.RowBuilder(53, np.dtype("float64"), kind, 20)
    for i in range(53):
        n = 0 if i % 11 == 0 else i % 17 + 1
        builder.resize_row(i, n // 2)
        builder.assign_row(i, np.arange(n // 2, dtype="float64") + 100 * i)
        builder.resize_row(i, n)
        for j in range(n // 2, n):
            builder[i, j] = i * 100 + j
    snapshot = builder.row(52)
    snapshot[:] = -1
    array = builder.commit()
    assert array[52, 1] == 5201
    for i in range(53):
        n = 0 if i % 11 == 0 else i % 17 + 1
        assert array.layout.row_size(i) == n
        assert np.array_equal(array[i].numpy(), np.arange(n) + 100 * i)
    with pytest.raises(RuntimeError):
        builder.commit()
    with pytest.raises(RuntimeError):
        builder.resize_row(0, 2)


def test_detached_edit_keeps_old_view_and_exclusive_edit_refuses_aliases():
    a = R.Array(R.RowLayout.ragged([5, 0, 7]))
    a[2] = np.arange(7)
    old_view = a[2]
    with pytest.raises(RuntimeError, match="unique"):
        a.edit_rows(exclusive=True)
    edit = a.edit_rows()
    edit.resize_row(0, 19)
    edit.resize_row(2, 3)
    a = edit.commit()
    assert old_view.numpy().tolist() == list(range(7))
    assert a[2].numpy().tolist() == [0, 1, 2]


def test_exclusive_commit_and_abandonment_are_explicit():
    a = R.Array(R.RowLayout.padded([5, 0, 7], 8))
    identity = a.identity
    edit = a.edit_rows(exclusive=True)
    assert not a.valid
    with pytest.raises(RuntimeError):
        a.alias()
    with pytest.raises(RuntimeError):
        a.read()
    edit.resize_row(1, 6)
    edit[1, 5] = 42
    assert edit.commit() is None
    assert a.valid and a.identity == identity and a.shape_version == 1
    assert a[1, 5] == 42
    del edit
    abandoned = a.edit_rows(exclusive=True)
    abandoned[0, 0] = 99
    del abandoned
    gc.collect()
    assert not a.valid
    with pytest.raises(RuntimeError):
        a.read()
    a.reset(R.RowLayout.dense(0, 5))
    assert a.valid


@pytest.mark.skipif(not R.cuda_enabled, reason="CUDA profile required")
def test_gpu_padded_metadata_and_detached_cpu_edit():
    host = R.Array(R.RowLayout.padded([0, 6, 7], 12))
    host[1] = np.arange(6)
    host[2] = np.arange(7) + 100
    gpu = host.to(R.Device.cuda(0))
    assert gpu.layout.sizes == [0, 6, 7]
    assert gpu[2, 6] == 106
    with pytest.raises(RuntimeError, match="host"):
        gpu.edit_rows(exclusive=True)
    draft = gpu.edit_rows()
    draft.resize_row(2, 9)
    draft[2, 8] = 88
    updated = draft.commit().to(R.Device.cuda(0))
    assert updated[2, 8] == 88
    assert gpu.layout.row_size(2) == 7
