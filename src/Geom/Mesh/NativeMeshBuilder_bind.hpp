#pragma once

#include "DNDS/Defines_bind.hpp"
#include "NativeMeshBuilder.hpp"
#include "Mesh.hpp"
#include "DNDS/Runtime/Array_bind.hpp"
#include <limits>

namespace DNDS::Geom
{
    inline void pybind11_NativeMeshBuilder_define(py::module_ &m)
    {
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
