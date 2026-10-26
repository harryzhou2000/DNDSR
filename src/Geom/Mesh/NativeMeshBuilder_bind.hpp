#pragma once

#include "DNDS/Defines_bind.hpp"
#include "NativeMeshBuilder.hpp"
#include "NativeMeshCGNS.hpp"
#include "NativeMeshTopology.hpp"
#include "NativeMeshRefinement.hpp"
#include "NativeMeshExtraction.hpp"
#include "FieldTransfer.hpp"
#include "Mesh.hpp"
#include "DNDS/Runtime/Array_bind.hpp"
#include <limits>
#include <set>

namespace DNDS::Geom
{
    inline void pybind11_NativeMeshBuilder_define(py::module_ &m)
    {
        auto periodicImport = [](const MPIInfo &mpi, py::object geometry, real tolerance)
        {
            std::optional<NativePeriodicImportOptions> result;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                DNDS_check_throw_info(!geometry.is_none() || tolerance == 1e-9, "periodic tolerance requires explicit periodic_geometry ({} selects defaults)");
                if (geometry.is_none() && tolerance == 1e-9) return;
                result.emplace();
                result->tolerance = tolerance;
                if (geometry.is_none()) return;
                DNDS_check_throw_info(py::isinstance<py::dict>(geometry), "periodic_geometry must be a dictionary or None");
                auto values = geometry.cast<py::dict>();
                const std::set<std::string> names{"translation1", "translation2", "translation3", "rotationCenter1", "rotationCenter2", "rotationCenter3", "eulerAngles1", "eulerAngles2", "eulerAngles3"};
                for (auto entry : values) DNDS_check_throw_info(names.count(py::cast<std::string>(entry.first)), "unknown periodic geometry option");
                for (int group = 1; group <= 3; ++group)
                {
                    auto get = [&](const std::string &name, const tPoint &fallback)
                    { return values.contains(name.c_str()) ? values[name.c_str()].cast<tPoint>() : fallback; };
                    result->frames.translation[size_t(group)].map() = get("translation" + std::to_string(group), result->frames.translation[size_t(group)].map());
                    result->frames.rotationCenter[size_t(group)].map() = get("rotationCenter" + std::to_string(group), tPoint::Zero());
                    const tPoint angles = get("eulerAngles" + std::to_string(group), tPoint::Zero());
                    result->frames.rotation[size_t(group)].map() = RotZ(angles[2]) * RotY(angles[1]) * RotX(angles[0]);
                } });
            return result;
        };
        m.def("native_mesh_requirements", [](const UnstructuredMesh &mesh)
              {
                  const auto requirements = mesh.CurrentNativeState()->Requirements();
                  py::dict result;
                  result["faces"] = requirements.faces;
                  result["edges"] = requirements.edges;
                  result["inverse"] = requirements.inverse;
                  result["cell_face_neighbors"] = requirements.cellFaceNeighbors;
                  result["wall_distance"] = requirements.wallDistance;
                  result["all_face_cells_local"] = requirements.allFaceCellsLocal;
                  return result; }, py::arg("mesh"), "Return the exact native publication groups, including empty published families.");
        m.def("extract_native_boundary", [](const UnstructuredMesh &mesh, int cellLayers, bool completeNodeStar)
              {
                  const auto mpi = mesh.mpi;
                  DNDS_check_throw_info(mpi.comm != MPI_COMM_NULL, "native boundary extraction requires a communicator");
                  py::gil_scoped_release release;
                  std::shared_ptr<const NativeMeshState> source;
                  FieldTransferDetail::LocalStage(mpi, [&] { source = mesh.CurrentNativeState(); });
                  auto result = NativeMeshExtraction::ExtractBoundary(*source, mpi, {cellLayers, completeNodeStar});
                  return UnstructuredMesh::FromNative(mpi, std::move(result.state)); }, py::arg("mesh"), py::arg("cell_layers") = 1, py::arg("complete_node_star") = false, "Collectively extract nonperiodic native boundary elements into a new mesh and namespace. Used nodes are compacted on their source owners; physical face order is preserved when present. Output has no boundary elements. Source queries and retained reads stay valid; owned Node->Node and Cell->Boundary ancestry is available through native_mesh_provenance. One-dimensional output supports line cells without point boundaries or facial topology.");
        auto derive = [](const UnstructuredMesh &mesh, int cellLayers, bool completeNodeStar, bool bisect)
        {
            const auto mpi = mesh.mpi;
            DNDS_check_throw_info(mpi.comm != MPI_COMM_NULL, "native refinement requires a communicator");
            py::gil_scoped_release release;
            std::shared_ptr<const NativeMeshState> source;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            { source = mesh.CurrentNativeState(); });
            const NativeMeshBuildOptions options{cellLayers, completeNodeStar};
            auto result = bisect ? NativeMeshRefinement::Bisect(*source, mpi, options)
                                 : NativeMeshRefinement::Elevate(*source, mpi, options);
            return UnstructuredMesh::FromNative(mpi, std::move(result.state));
        };
        m.def("elevate_native_mesh", [derive](const UnstructuredMesh &mesh, int cellLayers, bool completeNodeStar)
              { return derive(mesh, cellLayers, completeNodeStar, false); }, py::arg("mesh"), py::arg("cell_layers") = 1, py::arg("complete_node_star") = false, "Collectively elevate a nonperiodic O1 native mesh to O2 in a new namespace. Source buffers stay unchanged. Durable source persistent IDs are available through native_mesh_provenance.");
        m.def("bisect_native_mesh", [derive](const UnstructuredMesh &mesh, int cellLayers, bool completeNodeStar)
              { return derive(mesh, cellLayers, completeNodeStar, true); }, py::arg("mesh"), py::arg("cell_layers") = 1, py::arg("complete_node_star") = false, "Collectively split a nonperiodic O2 native mesh into O1 children in a new namespace. Durable source persistent IDs are available through native_mesh_provenance.");
        m.def("native_mesh_provenance", [](const UnstructuredMesh &mesh) -> py::object
              {
                  auto state = mesh.CurrentNativeState();
                  if (!state->Provenance())
                      return py::none();
                  const auto &provenance = *state->Provenance();
                  py::dict result, kinds, ids;
                  result["source_namespace"] = provenance.sourceNamespace;
                  for (const auto &[kind, sourceKind] : provenance.sourceKinds)
                      kinds[py::cast(kind)] = py::cast(sourceKind);
                  for (const auto &[kind, values] : provenance.sourceIds)
                  {
                      auto array = py::cast(Runtime::PythonArray::Share(values));
                      auto owner = py::capsule(new std::shared_ptr<const NativeMeshState>(state), [](void *pointer)
                      { delete static_cast<std::shared_ptr<const NativeMeshState> *>(pointer); });
                      // Registered pybind instances retain the capsule privately;
                      // each escaped array keeps the publication's read leases.
                      py::detail::keep_alive_impl(array, owner);
                      ids[py::cast(kind)] = std::move(array);
                  }
                  result["source_kinds"] = std::move(kinds);
                  result["source_ids"] = std::move(ids);
                  return result; }, py::arg("mesh"), "Return None or durable ancestry metadata: source_namespace, source_kinds and owned source_ids Runtime.Arrays. Arrays share storage and independently retain the immutable owner publication, so writes remain blocked after the mesh/dict is released. Runtime row() returns independent snapshots; retain native read leases for borrowed pointers.");
        m.def("read_native_cgns", [periodicImport](const MPIInfo &mpi, const std::string &path, int dimension, int readerRank, int cellLayers, bool completeNodeStar, py::object geometry, real tolerance, bool readerOwned)
              {
                  const auto periodic = periodicImport(mpi, geometry, tolerance);
                  AutoAppendName2ID boundaryIds;
                  py::gil_scoped_release release;
                  return UnstructuredMesh::FromNative(mpi, ReadNativeCGNS(mpi, path, dimension,
                      [&](const std::string &name) { return boundaryIds(name); }, readerRank, {cellLayers, completeNodeStar}, periodic, readerOwned ? NativeSerialDistribution::ReaderOwned : NativeSerialDistribution::Even)); }, py::arg("mpi"), py::arg("path"), py::arg("dimension"), py::arg("reader_rank") = 0, py::arg("cell_layers") = 1, py::arg("complete_node_star") = false, py::arg("periodic_geometry") = py::none(), py::arg("periodic_tolerance") = 1e-9, py::arg("reader_owned") = false, "Collectively import CGNS through native Runtime arrays. Optional periodic_geometry uses the SetPeriodicGeometry dictionary keys; radians and absolute matching tolerance. None with default tolerance disables identification; {} enables default transforms. BC main/donor labels determine pairing; CGNS periodic connection properties and same-original-node fixed points are rejected. All ranks supply identical options. Only reader_rank parses the file. reader_owned retains initial ownership there for serial RB partitioning.");
        m.def("read_native_cgns_with_names", [periodicImport](const MPIInfo &mpi, const std::string &path, int dimension, int readerRank, int cellLayers, bool completeNodeStar, py::object geometry, real tolerance, bool readerOwned)
              {
                  const auto periodic = periodicImport(mpi, geometry, tolerance);
                  auto names = std::make_shared<AutoAppendName2ID>();
                  std::shared_ptr<const NativeMeshState> state;
                  {
                      py::gil_scoped_release release;
                      state = ReadNativeCGNS(mpi, path, dimension,
                          [&](const std::string &name) { return (*names)(name); }, readerRank, {cellLayers, completeNodeStar}, periodic, readerOwned ? NativeSerialDistribution::ReaderOwned : NativeSerialDistribution::Even);
                      // Only readerRank parses names. Broadcast the exact assigned
                      // dictionary rather than recreating IDs in unordered order.
                      std::string encoded;
                      FieldTransferDetail::LocalStage(mpi, [&]
                      { if (mpi.rank == readerRank) encoded = nlohmann::json{{"names", names->n2id_map}, {"cap", names->id_cap}}.dump(); });
                      int count = 0;
                      FieldTransferDetail::LocalStage(mpi, [&]
                      { DNDS_check_throw_info(encoded.size() <= size_t(std::numeric_limits<int>::max()), "CGNS boundary dictionary exceeds MPI count range"); count = int(encoded.size()); });
                      MPI_Bcast(&count, 1, MPI_INT, readerRank, mpi.comm);
                      FieldTransferDetail::LocalStage(mpi, [&] { encoded.resize(size_t(count)); });
                      MPI_Bcast(encoded.data(), count, MPI_CHAR, readerRank, mpi.comm);
                      FieldTransferDetail::LocalStage(mpi, [&]
                      {
                          auto metadata = nlohmann::json::parse(encoded);
                          names->n2id_map = metadata.at("names").get<decltype(names->n2id_map)>();
                          names->id_cap = metadata.at("cap").get<t_index>();
                      });
                  }
                  return py::make_tuple(UnstructuredMesh::FromNative(mpi, std::move(state)), names); }, py::arg("mpi"), py::arg("path"), py::arg("dimension"), py::arg("reader_rank") = 0, py::arg("cell_layers") = 1, py::arg("complete_node_star") = false, py::arg("periodic_geometry") = py::none(), py::arg("periodic_tolerance") = 1e-9, py::arg("reader_owned") = false, "Collective native CGNS import returning (mesh, boundary name dictionary). Optional periodic_geometry and periodic_tolerance match read_native_cgns. The exact reader-assigned names and IDs are retained on every rank.");
        m.def("with_native_topology", [](const UnstructuredMesh &mesh, bool faces, bool edges, int ghostLayers)
              {
                  const auto mpi = mesh.mpi;
                  DNDS_check_throw_info(mpi.comm != MPI_COMM_NULL, "native topology requires a communicator");
                  std::shared_ptr<const NativeMeshState> source;
                  FieldTransferDetail::LocalStage(mpi, [&] { source = mesh.CurrentNativeState(); });
                  py::gil_scoped_release release;
                  return UnstructuredMesh::FromNative(mpi, NativeMeshTopology::Build(*source, mpi, faces, edges, ghostLayers)); }, py::arg("mesh"), py::arg("faces") = true, py::arg("edges") = false, py::arg("ghost_layers") = 1, "Collectively return a new native mesh with face/edge topology. Source must be a nonperiodic native publication; source storage and escaped read views remain valid. Owned cell face/edge cones are local. Outer ghost cell cones can contain unavailable negative global references; complete face-to-cell closure does not imply complete converse ghost cell cones.");
        m.def("build_native_mesh", [](const MPIInfo &mpi, int dimension, py::object coordinates, py::object cellNodes, py::object cellInfo, py::object boundaryNodes, py::object boundaryInfo, int cellLayers, bool completeNodeStar, py::object nodeIds, py::object cellIds, py::object boundaryIds)
              {
            DNDS_check_throw_info(mpi.comm != MPI_COMM_NULL, "native Python mesh construction requires a communicator");
            struct OwnedInput
            {
                Runtime::Array<real> coordinates;
                Runtime::Array<index> cellNodes, boundaryNodes;
                Runtime::Array<ElemInfo> cellInfo, boundaryInfo;
                std::map<EntityKind, Runtime::Array<index>> ids;
            };
            std::optional<OwnedInput> owned;
            std::string detail;
            try
            {
                int actualRank = -1, actualSize = 0;
                MPI_Comm_rank(mpi.comm, &actualRank);
                MPI_Comm_size(mpi.comm, &actualSize);
                DNDS_check_throw_info(mpi.rank == actualRank && mpi.size == actualSize,
                                      "native Python mesh MPIInfo differs from its communicator");
                auto convert = [](py::handle object, auto *type)
                {
                    using T = std::remove_pointer_t<decltype(type)>;
                    DNDS_check_throw_info(py::isinstance<Runtime::PythonArray>(object),
                                          "native mesh inputs must be Runtime.Array handles");
                    const auto &array = py::cast<const Runtime::PythonArray &>(object);
                    auto native = array.template AsArray<T>();
                    DNDS_check_throw_info(native.GetDevice() == Runtime::Device::Host() &&
                                              native.Rows() <= size_t(std::numeric_limits<index>::max()),
                                          "native Python mesh inputs require host storage and supported row counts");
                    auto read = native.Read();
                    return native;
                };
                OwnedInput values{
                    convert(coordinates, static_cast<real *>(nullptr)),
                    convert(cellNodes, static_cast<index *>(nullptr)),
                    convert(boundaryNodes, static_cast<index *>(nullptr)),
                    convert(cellInfo, static_cast<ElemInfo *>(nullptr)),
                    convert(boundaryInfo, static_cast<ElemInfo *>(nullptr)), {}};
                auto width = [](const auto &array, size_t expected)
                {
                    const auto shape = array.Layout();
                    for (size_t row = 0; row < shape.Rows(); ++row)
                        DNDS_check_throw_info(shape.RowSize(row) == expected, "native Python mesh input row width mismatch");
                };
                width(values.coordinates, 3);
                width(values.cellInfo, 1);
                width(values.boundaryInfo, 1);
                DNDS_check_throw_info(values.cellNodes.Rows() == values.cellInfo.Rows() &&
                                          values.boundaryNodes.Rows() == values.boundaryInfo.Rows(),
                                      "native Python element information and cone row counts differ");
                auto optionalIds = [&](py::handle object, EntityKind kind, size_t count)
                {
                    if (object.is_none()) return;
                    auto ids = convert(object, static_cast<index *>(nullptr));
                    width(ids, 1);
                    DNDS_check_throw_info(!ids.Layout().IsVariable() && ids.Layout().Width() == 1 && ids.Rows() == count,
                                          "native Python persistent IDs require one dense owned column");
                    values.ids.emplace(kind, std::move(ids));
                };
                optionalIds(nodeIds, EntityKind::Node, values.coordinates.Rows());
                optionalIds(cellIds, EntityKind::Cell, values.cellNodes.Rows());
                optionalIds(boundaryIds, EntityKind::Bnd, values.boundaryNodes.Rows());
                owned.emplace(std::move(values));
            }
            catch (const std::exception &error)
            {
                detail = error.what();
            }
            catch (...)
            {
                detail = "unknown Python array conversion error";
            }
            int valid = bool(owned);
            MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, mpi.comm);
            DNDS_check_throw_info(valid, "native Python mesh input validation failed on a rank" +
                                            (detail.empty() ? std::string{} : "; " + detail));
            py::gil_scoped_release release;
            auto nodes = EntitySpace::CreateOwned(EntityKind::Node, mpi, index(owned->coordinates.Rows()));
            auto cells = nodes.CreateSibling(EntityKind::Cell, index(owned->cellNodes.Rows()));
            auto boundaries = nodes.CreateSibling(EntityKind::Bnd, index(owned->boundaryNodes.Rows()));
            NativeMeshInput input{mpi, dimension, cells, nodes, boundaries,
                                   std::move(owned->cellNodes), std::move(owned->boundaryNodes),
                                   std::move(owned->coordinates), std::move(owned->cellInfo),
                                   std::move(owned->boundaryInfo), std::move(owned->ids)};
            NativeMeshBuildOptions options;
            options.cellLayers = cellLayers;
            options.completeNodeStar = completeNodeStar;
            return UnstructuredMesh::FromNative(mpi, NativeMeshBuilder::Build(std::move(input), options)); }, py::arg("mpi"), py::arg("dimension"), py::arg("coordinates"), py::arg("cell_nodes"), py::arg("cell_info"), py::arg("boundary_nodes"), py::arg("boundary_info"), py::arg("cell_layers") = 1, py::arg("complete_node_star") = false, py::arg("node_ids") = py::none(), py::arg("cell_ids") = py::none(), py::arg("boundary_ids") = py::none(), "Collectively build a nonperiodic native primary mesh from owned host Runtime.Arrays. Connectivity uses rank-offset global node numbers; coordinates are float64, connectivity/IDs int64, metadata ElemInfo. Coordinates/metadata accept dense, padded or uniform CSR rows; uniform metadata normalization shares the original value buffer. Input backing buffers stay read-protected for the publication lifetime; retain read scopes for escaped rows.");
    }
}
