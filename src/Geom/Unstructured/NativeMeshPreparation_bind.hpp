#pragma once
#include "Geom/Unstructured/NativeMeshPreparation.hpp"
#include "Geom/Compat/NativeMeshLegacyArchive.hpp"
#include "Geom/Unstructured/UnstructuredMeshHelpers.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "Geom/Transition/FieldTransfer.hpp"
#include "DNDS/Defines_bind.hpp"

namespace DNDS::Geom
{
    inline void pybind11_NativeMeshPreparation_define(py::module_ &m)
    {
        m.def("prepare_native_mesh_setup", [](ssp<UnstructuredMesh> mesh, int parts, int innerParts, int layers, bool edges, bool inPlace)
              {
            const auto mpi = mesh->mpi;
            py::gil_scoped_release release;
            std::array<int, 5> settings{parts, innerParts, layers, int(edges), int(inPlace)}, expected = settings;
            MPI_Bcast(expected.data(), int(expected.size()), MPI_INT, 0, mpi.comm);
            FieldTransferDetail::LocalStage(mpi, [&]
                                            { DNDS_check_throw_info(settings == expected && parts >= 0 && innerParts > 0 && layers > 0,
                                                                  "native setup options differ across ranks or invalid counts"); });
            auto target = inPlace ? mesh : UnstructuredMesh::FromNative(mpi, mesh->CurrentNativeState());
            UnstructuredMeshSerialRW output(target, 0);
            PrepareMeshOptions options;
            options.reorderCells = parts != 0;
            options.reorderParts = parts;
            options.reorderPartsInner = innerParts;
            options.ghostLayers = layers;
            options.buildEdges = edges;
            options.buildSerialOut = false;
            PrepareMesh(*target, output, options);
            return target; }, py::arg("mesh"), py::arg("reorder_parts") = 0, py::arg("reorder_inner_parts") = 1, py::arg("ghost_layers") = 1, py::arg("edges") = false, py::arg("in_place") = false, "Collective owned setup: optional cell permutation precedes face/edge construction; final assembly runs once with complete owned-node stars and retained custom paths. Zero reorder_parts disables permutation. Returns a detached facade by default; in_place updates the original facade and rejects active reads. Keep the returned mesh; allocate external solver fields after setup. All ranks supply identical options.");
        m.def("read_native_archive_file", [](const MPIInfo &mpi, Serializer::SerializerFactory factory, const std::string &path, int ghostLayers, py::object partitionOptions)
              {
            const bool repartition = !partitionOptions.is_none();
            PartitionOptions options;
            FieldTransferDetail::LocalStage(mpi, [&]
            {
                if (!repartition) return;
                DNDS_check_throw_info(py::isinstance<py::dict>(partitionOptions), "archive partition_options must be a dictionary or None");
                for (const auto &entry : partitionOptions.cast<py::dict>())
                {
                    const auto key = py::cast<std::string>(entry.first);
                    if (key == "metisType")
                    {
                        DNDS_check_throw_info(py::isinstance<py::str>(entry.second), "archive partition method must be a string");
                        options.metisType = py::cast<std::string>(entry.second);
                        continue;
                    }
                    DNDS_check_throw_info(key == "metisUfactor" || key == "metisSeed" || key == "edgeWeightMethod" || key == "metisNcuts", "unknown archive partition option");
                    DNDS_check_throw_info(py::isinstance<py::int_>(entry.second) && !py::isinstance<py::bool_>(entry.second), "archive partition option must be an integer");
                    const int value = py::cast<int>(entry.second);
                    if (key == "metisUfactor") options.metisUfactor = value;
                    else if (key == "metisSeed") options.metisSeed = value;
                    else if (key == "edgeWeightMethod") options.edgeWeightMethod = value;
                    else options.metisNcuts = value;
                }
                DNDS_check_throw_info((options.metisType == "KWAY" || options.metisType == "RB") &&
                    options.metisUfactor > 0 && options.metisSeed >= 0 && options.metisNcuts > 0 &&
                    (options.edgeWeightMethod == 0 || options.edgeWeightMethod == 1) &&
                    options.metisSeed <= std::numeric_limits<int>::max() - (options.metisNcuts - 1),
                    "archive native partition options invalid");
            });
            py::gil_scoped_release release;
            return ReadNativeMeshFromArchive(mpi, std::move(factory), path, repartition, options, ghostLayers); }, py::arg("mpi"), py::arg("factory"), py::arg("path"), py::arg("ghost_layers") = 1, py::arg("partition_options") = py::none(), "Collectively open/decode/close native-full or legacy-primary archives. None preserves matching ownership. A partition dictionary explicitly requests H5 even-split loading (including changed MPI size) followed by native graph partitioning; per-rank JSON cannot change MPI size.");
        m.def("read_compatible_native_archive", [](const MPIInfo &mpi, Serializer::SerializerBaseSSP serializer, const std::string &name, Serializer::ArrayGlobalOffset partition, int layers)
              {
            py::gil_scoped_release release;
            return UnstructuredMesh::FromNative(mpi, ReadCompatibleNativeMeshArchive(mpi, std::move(serializer), name, partition, layers)); }, py::arg("mpi"), py::arg("serializer"), py::arg("name"), py::arg("partition") = Serializer::ArrayGlobalOffset_Unknown, py::arg("ghost_layers") = 1, "Collectively decode exactly one native-full or legacy-primary archive schema directly into native storage. Legacy import creates a fresh namespace and preserves original IDs.");
        m.def("prepare_native_topology", [](const UnstructuredMesh &mesh, bool faces, bool edges, int ghostLayers)
              {
            const auto mpi = mesh.mpi;
            std::shared_ptr<const UnstructuredMeshState> source;
            FieldTransferDetail::LocalStage(mpi, [&] { source = mesh.CurrentNativeState(); });
            py::gil_scoped_release release;
            return UnstructuredMesh::FromNative(mpi, NativeMeshPreparation::PrepareTopology(*source, mpi, faces, edges, ghostLayers)); }, py::arg("mesh"), py::arg("faces") = true, py::arg("edges") = false, py::arg("ghost_layers") = 1, "Collectively complete missing native topology and refresh closure. Existing families/IDs are retained, including empty families.");
        m.def("partition_native_mesh", [](const UnstructuredMesh &mesh, const std::string &method, int ufactor, int seed, int edgeWeight, int cuts, int ghostLayers, bool deferFaces)
              {
            const auto mpi = mesh.mpi;
            std::shared_ptr<const UnstructuredMeshState> source;
            FieldTransferDetail::LocalStage(mpi, [&] { source = mesh.CurrentNativeState(); });
            const NativePartitionOptions options{method, ufactor, seed, edgeWeight, cuts, ghostLayers, deferFaces};
            py::gil_scoped_release release;
            auto result = NativeMeshPreparation::Partition(*source, mpi, options);
            return UnstructuredMesh::FromNative(mpi, std::move(result.state)); }, py::arg("mesh"), py::arg("metisType") = "KWAY", py::arg("metisUfactor") = 20, py::arg("metisSeed") = 0, py::arg("edgeWeightMethod") = 0, py::arg("metisNcuts") = 3, py::arg("ghost_layers") = 1, py::arg("defer_new_faces_until_preparation") = false, "Collectively return a native facial-graph partition. KWAY accepts distributed ownership; RB requires all cells on one rank (reader_owned ingress). Weight 0 is uniform; weight 1 uses physical face-node diameter with integer-overflow-safe scaling. Deferral retains primary input only for unweighted reader-owned meshes with no existing faces; call prepare_native_mesh afterward. Other paths retain their facial behavior. External fields remain associated with the source; prepare before field allocation, or use C++ Partition's returned transfers.");
        m.def("reorder_native_mesh", [](const UnstructuredMesh &mesh, int parts, int innerParts, int ghostLayers)
              {
            const auto mpi = mesh.mpi;
            std::shared_ptr<const UnstructuredMeshState> source;
            FieldTransferDetail::LocalStage(mpi, [&] { source = mesh.CurrentNativeState(); });
            py::gil_scoped_release release;
            auto result = NativeMeshTransition::ReorderLocalCells(*source, mpi, parts, innerParts, ghostLayers);
            return UnstructuredMesh::FromNative(mpi, std::move(result.state)); }, py::arg("mesh"), py::arg("n_parts") = 1, py::arg("n_parts_inner") = 1, py::arg("ghost_layers") = 1, "Collectively return a detached native local cell permutation. Prepare before allocating external fields; C++ callers can explicitly apply all returned family transfers.");
    }
}
