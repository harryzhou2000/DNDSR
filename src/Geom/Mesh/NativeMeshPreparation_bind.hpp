#pragma once
#include "NativeMeshPreparation.hpp"
#include "NativeMeshLegacyArchive.hpp"
#include "Mesh.hpp"
#include "FieldTransfer.hpp"
#include "DNDS/Defines_bind.hpp"

namespace DNDS::Geom
{
    inline void pybind11_NativeMeshPreparation_define(py::module_ &m)
    {
        m.def("read_compatible_native_archive", [](const MPIInfo &mpi, Serializer::SerializerBaseSSP serializer, const std::string &name, Serializer::ArrayGlobalOffset partition, int layers)
              {
            py::gil_scoped_release release;
            return UnstructuredMesh::FromNative(mpi, ReadCompatibleNativeMeshArchive(mpi, std::move(serializer), name, partition, layers)); }, py::arg("mpi"), py::arg("serializer"), py::arg("name"), py::arg("partition") = Serializer::ArrayGlobalOffset_Unknown, py::arg("ghost_layers") = 1, "Collectively decode exactly one native-full or legacy-primary archive schema directly into native storage. Legacy import creates a fresh namespace and preserves original IDs.");
        m.def("prepare_native_topology", [](const UnstructuredMesh &mesh, bool faces, bool edges, int ghostLayers)
              {
            const auto mpi = mesh.mpi;
            std::shared_ptr<const NativeMeshState> source;
            FieldTransferDetail::LocalStage(mpi, [&] { source = mesh.CurrentNativeState(); });
            py::gil_scoped_release release;
            return UnstructuredMesh::FromNative(mpi, NativeMeshPreparation::PrepareTopology(*source, mpi, faces, edges, ghostLayers)); }, py::arg("mesh"), py::arg("faces") = true, py::arg("edges") = false, py::arg("ghost_layers") = 1, "Collectively complete missing native topology and refresh closure. Existing families/IDs are retained, including empty families.");
        m.def("partition_native_mesh", [](const UnstructuredMesh &mesh, const std::string &method, int ufactor, int seed, int edgeWeight, int cuts, int ghostLayers)
              {
            const auto mpi = mesh.mpi;
            std::shared_ptr<const NativeMeshState> source;
            FieldTransferDetail::LocalStage(mpi, [&] { source = mesh.CurrentNativeState(); });
            const NativePartitionOptions options{method, ufactor, seed, edgeWeight, cuts, ghostLayers};
            py::gil_scoped_release release;
            auto result = NativeMeshPreparation::Partition(*source, mpi, options);
            return UnstructuredMesh::FromNative(mpi, std::move(result.state)); }, py::arg("mesh"), py::arg("metisType") = "KWAY", py::arg("metisUfactor") = 20, py::arg("metisSeed") = 0, py::arg("edgeWeightMethod") = 0, py::arg("metisNcuts") = 3, py::arg("ghost_layers") = 1, "Collectively return a native facial-graph partition. KWAY/unweighted only; unsupported policies reject. External fields remain associated with the source; prepare before field allocation, or use C++ Partition's returned transfers.");
        m.def("reorder_native_mesh", [](const UnstructuredMesh &mesh, int parts, int innerParts, int ghostLayers)
              {
            const auto mpi = mesh.mpi;
            std::shared_ptr<const NativeMeshState> source;
            FieldTransferDetail::LocalStage(mpi, [&] { source = mesh.CurrentNativeState(); });
            py::gil_scoped_release release;
            auto result = NativeMeshTransition::ReorderLocalCells(*source, mpi, parts, innerParts, ghostLayers);
            return UnstructuredMesh::FromNative(mpi, std::move(result.state)); }, py::arg("mesh"), py::arg("n_parts") = 1, py::arg("n_parts_inner") = 1, py::arg("ghost_layers") = 1, "Collectively return a detached native local cell permutation. Prepare before allocating external fields; C++ callers can explicitly apply all returned family transfers.");
    }
}
