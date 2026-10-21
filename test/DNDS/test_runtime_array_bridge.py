"""Actual typed C++/Python sharing through a separately compiled extension.

Build runtime_array_bridge_test and add build/test/cpp to PYTHONPATH, after
building AND installing the four production Python modules for that profile.
"""

import gc
import importlib.util
import os
import numpy as np
import pytest
from DNDSR.DNDS import Runtime as R
from DNDSR import DNDS
from mpi4py import MPI

if importlib.util.find_spec("runtime_array_bridge_test") is None:
    pytest.skip("build runtime_array_bridge_test and include build/test/cpp in PYTHONPATH", allow_module_level=True)
# A present but broken extension must fail collection, not silently skip coverage.
import runtime_array_bridge_test as bridge


@pytest.fixture(params=[R.Device.host(), R.Device.cuda(0)])
def device(request):
    value = request.param
    if value.space == R.MemorySpace.CUDA and not R.cuda_enabled:
        pytest.skip("this Python build has no CUDA backend")
    return value


def test_native_owner_and_python_share_values_layout_and_controller(device):
    owner = bridge.NativeOwner(device)
    state = owner.state
    assert isinstance(state, R.Array)
    assert state.shape == (3, 5)
    assert state.layout.stride == 8
    assert owner.shares_state(state)
    assert owner.shares_ids(owner.ids)  # Record, CPU CSR offsets and CUDA offsets.
    state[:] = np.arange(15).reshape(3, 5)
    assert owner.read_state(2, 3) == 13
    before = state.version
    owner.scale(2)
    assert state.version == before + 1
    assert state[2, 3] == 26  # CUDA profile executes a real typed C++ CUDA kernel.
    with state.read() as read:
        with owner.state.read() as native_read:
            assert read.unsafe_address() == native_read.unsafe_address()
        with pytest.raises(RuntimeError):
            owner.scale(3)

    def during_cpp_read():
        assert state[2, 3] == 26
        with pytest.raises(RuntimeError):
            state[0, 0] = 3

    def during_cpp_write():
        with pytest.raises(RuntimeError):
            state[0, 0]

    bridge.with_read(state, during_cpp_read)
    bridge.with_write(state, during_cpp_write)
    alias = bridge.roundtrip(state)
    assert alias.identity == state.identity
    assert owner.shares_state(alias)
    bridge.poison(alias)
    assert not state.valid and not owner.state.valid


def test_foreign_owner_survives_python_slices_and_cpp_aliases():
    array, watch = bridge.external_state()
    assert array.shape == (53, 7) and array.layout.stride == 8
    array[:] = np.arange(53 * 7).reshape(53, 7)
    native_alias = bridge.roundtrip(array)
    retained = array[::3, ::-2]
    del array
    gc.collect()
    assert watch.alive
    native_alias[51, 6] = 900
    assert retained[-1, 0] == 900
    del native_alias
    gc.collect()
    assert watch.alive
    snapshot = retained.numpy()
    del retained
    gc.collect()
    assert not watch.alive
    assert snapshot[-1, 0] == 900


def test_bound_pod_codec_is_reused_by_row_construction_and_editing():
    draft = R.RowBuilder(5, bridge.ClassMetadata, R.RowLayoutKind.Padded, 4)
    draft.resize_row(2, 3)
    draft[2, 1] = bridge.ClassMetadata(73, 0.5)
    value = draft[2, 1]
    assert isinstance(value, bridge.ClassMetadata)
    assert value.tag == 73
    array = draft.commit()
    assert array.element_type is bridge.ClassMetadata
    assert array[2, 1].weight == 0.5
    old = array[2]
    with pytest.raises(RuntimeError, match="unique"):
        array.edit_rows(exclusive=True)
    detached = array.edit_rows()
    detached.resize_row(2, 4)
    detached[2, 3] = bridge.ClassMetadata(91, 1.25)
    array = detached.commit()
    assert array[2, 3].tag == 91
    assert old[1].tag == 73
    edit = array.edit_rows(exclusive=True)
    edit[2, 3] = bridge.ClassMetadata(92, 1.5)
    edit.commit()
    assert array[2, 3].weight == 1.5


def test_python_created_array_can_be_used_and_retained_by_native_owner(device):
    owner = bridge.NativeOwner(device)
    state = R.Array(R.RowLayout.dense(2, 5, 7), np.dtype("float64"), device)
    state[:] = 3
    owner.replace_state(state)
    assert owner.shares_state(state)
    bridge.scale(state, 4)
    assert owner.read_state(1, 4) == 12
    owner.scale(0.5)
    assert state[1, 4] == 6
    identity = state.identity
    del state
    gc.collect()
    assert owner.state.identity == identity
    assert owner.read_state(1, 4) == 6


def test_ragged_integer_and_custom_pod_cross_language(device):
    owner = bridge.NativeOwner(device)
    ids = owner.ids
    ids[1] = [11, 12]
    ids[2] = [21, 22, 23]
    assert owner.read_id(2, 2) == 23
    alias = bridge.roundtrip_ids(ids)
    assert owner.shares_ids(alias)
    alias[1, 0] = -7
    assert owner.read_id(1, 0) == -7
    metadata = owner.metadata
    assert metadata.dtype == bridge.metadata_dtype
    metadata[0, 0] = (17, 2.5)
    assert owner.read_metadata(0, 0) == (17, 2.5)
    alias_meta = bridge.roundtrip_metadata(metadata)
    assert alias_meta.identity == metadata.identity
    alias_meta[2, 1] = (91, 0.25)
    assert owner.read_metadata(2, 1) == (91, 0.25)


def test_exports_retain_allocations_and_metadata_after_cpp_reset_or_destruction(device):
    owner = bridge.NativeOwner(device)
    old = owner.state
    old[:] = 9
    ids = owner.ids
    ids[2] = [1, 2, 3]
    owner.reset_state()
    assert owner.state.identity != old.identity
    assert old[2, 4] == 9
    assert owner.read_state(0, 0) == 0
    del owner
    gc.collect()
    assert ids[2, 2] == 3
    with ids.read() as read:
        ids.reset(R.RowLayout.dense(0, 0), device)
        assert read[2, 2] == 3
    bridge.scale(old, 2)
    assert old[2, 4] == 18


def test_nested_slice_and_access_retain_storage_until_last_owner(device):
    owner = bridge.NativeOwner(device)
    array = owner.ids
    array[2] = [11, 12, 13]
    watch = bridge.StorageWatch(array)
    row = array[2]
    nested = row[::-1]
    array.reset(R.RowLayout.dense(0, 0), device)
    del owner, array, row
    gc.collect()
    assert watch.storage_alive and watch.layout_alive
    assert nested.tolist() == [13, 12, 11]
    # The retained view itself holds no lease, so a write is still permitted.
    nested[1] = 42
    snapshot = nested.numpy()
    access = nested._base.read()
    del nested
    gc.collect()
    assert watch.storage_alive and watch.layout_alive
    assert access[2, 1] == 42  # Original CSR metadata, including device offsets.
    access.wait()
    with pytest.raises(RuntimeError):
        access[2, 1]
    # A closed Python access still owns its metadata until its object is freed.
    del access
    gc.collect()
    assert not watch.storage_alive and not watch.layout_alive
    assert snapshot.tolist() == [13, 42, 11]  # NumPy owns independent storage.


def test_static_width_conversion_needs_no_python_class_instantiation():
    a = bridge.make_fixed37()
    b = bridge.make_fixed113()
    assert type(a) is R.Array and type(b) is R.Array
    assert a.shape == (2, 37) and b.shape == (1, 113)
    a[1, 36] = 7.5
    b[0, 112] = 91
    assert bridge.roundtrip(a)[1, 36] == 7.5
    assert bridge.roundtrip_ids(b)[0, 112] == 91
    with pytest.raises(RuntimeError, match="fixed row width"):
        bridge.scale(a, 2)
    with pytest.raises(TypeError):
        bridge.scale(R.Array(R.RowLayout.dense(3, 5), np.dtype("int64")), 2)
    with pytest.raises(RuntimeError, match="fixed row width"):
        bridge.scale(R.Array(R.RowLayout.ragged([5, 5]), np.dtype("float64")), 2)
    # Same object size does not make a different struct layout compatible.
    wrong_pod = np.dtype([("other", "i8"), ("weight", "f8")])
    with pytest.raises(TypeError):
        bridge.roundtrip_metadata(R.Array(R.RowLayout.dense(1, 1), wrong_pod))


def test_bound_pod_class_snapshots_and_tracked_live_fields(device):
    owner = bridge.NativeOwner(device)
    a = owner.class_metadata
    assert a.element_type is bridge.ClassMetadata
    a[0, 0] = bridge.ClassMetadata(13, 2.5)
    value = a[0, 0]
    assert isinstance(value, bridge.ClassMetadata)
    value.tag = 99  # Class values are snapshots, not unleased mutable references.
    assert owner.read_class_metadata(0, 0).tag == 13
    a[0, 0] = value
    assert owner.read_class_metadata(0, 0).tag == 99
    live = a.element(0, 0)
    before = a.version
    live.tag = 17
    assert a.version == before + 1
    live.scale(3)
    assert owner.read_class_metadata(0, 0).weight == 7.5
    with a.read():
        assert live.tag == 17
        with pytest.raises(RuntimeError):
            live.tag = 18
    before = a.version
    with pytest.raises(AttributeError):
        live.no_such_field = 1
    assert a.valid and a.version == before

    def failed_edit(value):
        value.tag = 123
        raise ValueError("cancel snapshot")

    with pytest.raises(ValueError, match="cancel snapshot"):
        live.edit(failed_edit)
    assert live.tag == 17 and a.valid and a.version == before
    with a.write() as edit:
        value = edit[0, 0]
        value.weight = 4.25
        edit[0, 0] = value
    assert live.weight == 4.25
    alias = bridge.roundtrip_class_metadata(a)
    assert alias.identity == a.identity and alias.element_type is bridge.ClassMetadata
    clone = a.clone()
    assert clone.element_type is bridge.ClassMetadata
    assert clone[0, 0].tag == 17
    del owner, a, alias
    gc.collect()
    live.tag = 41
    assert live.snapshot().tag == 41
    assert clone[0, 0].tag == 17


def test_python_constructed_class_pod_and_type_rejection(device):
    a = R.Array(R.RowLayout.dense(2, 3), bridge.ClassMetadata, device)
    a[:] = bridge.ClassMetadata(5, 1.5)
    assert bridge.roundtrip_class_metadata(a)[1, 2].weight == 1.5
    assert a.element(1, 2).tag == 5
    a[0][1] = bridge.ClassMetadata(9, 2.0)
    assert a[0, 1].tag == 9
    before = a.version
    with pytest.raises((TypeError, RuntimeError)):
        a[0, 0] = object()
    assert a.valid and a.version == before
    untyped = R.Array(a.layout, a.dtype, device)
    with pytest.raises(TypeError):
        bridge.roundtrip_class_metadata(untyped)
    with pytest.raises(RuntimeError):
        a.copy_from(untyped)
    retained = a.element(0, 1)
    a.reset(R.RowLayout.dense(1, 1), device)
    assert a.element_type is bridge.ClassMetadata
    assert retained.tag == 9


def test_pod_view_proxy_and_saved_method_ownership(device):
    owner = bridge.NativeOwner(device)
    array = owner.class_metadata
    array[2, 1] = bridge.ClassMetadata(29, 3)
    watch = bridge.StorageWatch(array)
    row = array[2]
    live = array.element(2, 1)
    saved_method = live.scale
    del owner, array, live
    gc.collect()
    assert watch.storage_alive and watch.layout_alive
    saved_method(2)
    snapshot = row[1]
    assert isinstance(snapshot, bridge.ClassMetadata) and snapshot.weight == 6
    del row, saved_method
    gc.collect()
    assert not watch.storage_alive and not watch.layout_alive
    snapshot.tag = 30  # A class value owns its own POD copy.
    assert snapshot.tag == 30


def test_pod_edit_retains_original_storage_if_callback_rebinds_handle(device):
    array = R.Array(R.RowLayout.ragged([0, 2]), bridge.ClassMetadata, device)
    array[1, 1] = bridge.ClassMetadata(7, 1)
    original = array.alias()

    def edit(value):
        array.reset(R.RowLayout.dense(0, 0), device)
        value.tag = 8

    array._edit_object(1, 1, edit)
    assert array.shape == (0, 0)
    assert original[1, 1].tag == 8


@pytest.mark.mpi
def test_class_pod_transfer_and_checkpoint(device):
    comm = MPI.COMM_WORLD
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    a = R.Array(R.RowLayout.ragged([1, 0, 2]), bridge.ClassMetadata, device)
    a[0, 0] = bridge.ClassMetadata(comm.rank, 0.5)
    a[2, 1] = bridge.ClassMetadata(comm.rank + 100, 1.5)
    route = R.RowTransferPlan(mpi, 3, 2, [R.RowRequest((comm.rank + 1) % comm.size, 0, 0),
                                         R.RowRequest(comm.rank, 2, 1)])
    b = R.Array(route.destination_layout(a.layout), bridge.ClassMetadata, device)
    R.PayloadPlan(route, a.layout, b.layout, a.element_schema()).bind(a, b).begin(R.ExecutionContext(device)).wait()
    assert b[0, 0].tag == (comm.rank + 1) % comm.size
    assert bridge.roundtrip_class_metadata(b)[1, 1].tag == comm.rank + 100
    name = comm.bcast(f"/tmp/dnds_class_pod_{os.getpid()}.h5" if comm.rank == 0 else None)
    serializer = DNDS.Serializer.SerializerH5(mpi)
    serializer.OpenFile(name, False)
    b.write_serializer(serializer, "class_field", mpi)
    serializer.CloseFile()
    serializer.OpenFile(name, True)
    c = R.Array.read_serializer(serializer, "class_field", mpi, dtype=bridge.ClassMetadata, device=device)
    serializer.CloseFile()
    assert c.element_type is bridge.ClassMetadata
    assert bridge.roundtrip_class_metadata(c)[1, 1].tag == comm.rank + 100
    comm.Barrier()
    if comm.rank == 0:
        os.remove(name)
