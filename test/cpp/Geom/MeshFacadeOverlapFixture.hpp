#pragma once

#include "Geom/Mesh/Mesh.hpp"
#include "Geom/Mesh/EntitySpace.hpp"
#include "DNDS/Runtime/EagerArray.hpp"
#include "DNDS/Runtime/MPIProgress.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <map>
#include <vector>

namespace DNDS::Geom::OverlapTest
{
    using Index = DNDS::index;
    namespace R = DNDS::Runtime;

    struct Fixture
    {
        ssp<UnstructuredMesh> mesh;
        std::vector<real> expected;
    };

    /// Nonlinear, exactly representable values avoid a linear field's zero
    /// interior Laplacian accidentally hiding a missing core computation.
    inline real ManufacturedValue(Index original)
    {
        const auto id = uint64_t(original);
        return 0.25 + real((37 * id + 11 * id * id) % 101);
    }

    /// Serial oracle uses CGNS node IDs and independently matches polygon edges.
    /// No distributed adjacency, ghost map or facade query enters this result.
    inline Fixture BuildFixture(MPIInfo mpi)
    {
        Fixture result;
        result.mesh = make_ssp<UnstructuredMesh>(mpi, 2);
        auto &mesh = *result.mesh;
        UnstructuredMeshSerialRW reader(result.mesh, 0);
        const auto path = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path() /
                          "data" / "mesh" / "UniformSquare_10.cgns";
        DNDS_check_throw_info(std::filesystem::exists(path), "overlap CGNS fixture is missing");
        reader.ReadFromCGNSSerial(path.string());
        if (mpi.rank == 0)
        {
            struct Edge
            {
                Index cell, a, b;
                real length;
            };
            std::map<std::pair<Index, Index>, Edge> edges;
            result.expected.resize(size_t(reader.cell2nodeSerial->Size()), 0);
            for (Index cell = 0; cell < reader.cell2nodeSerial->Size(); ++cell)
            {
                const auto nodes = (*reader.cell2nodeSerial)[cell];
                const auto type = (*reader.cellElemInfoSerial)(cell, 0).getElemType();
                DNDS_check_throw_info(type == Elem::Tri3 || type == Elem::Quad4, "overlap oracle requires linear polygons");
                for (size_t slot = 0; slot < nodes.size(); ++slot)
                {
                    const auto a = nodes[slot], b = nodes[(slot + 1) % nodes.size()];
                    const auto delta = ((*reader.coordSerial)[a] - (*reader.coordSerial)[b]).eval();
                    const real length = delta.norm();
                    const auto key = std::minmax(a, b);
                    auto inserted = edges.emplace(std::make_pair(key.first, key.second), Edge{cell, a, b, length});
                    if (!inserted.second)
                    {
                        const auto &other = inserted.first->second;
                        DNDS_check_throw_info(std::abs(other.length - length) < 1e-12, "serial shared-edge geometry differs");
                        const real flux = length * (ManufacturedValue(other.cell) - ManufacturedValue(cell));
                        result.expected[size_t(cell)] += flux;
                        result.expected[size_t(other.cell)] -= flux;
                        edges.erase(inserted.first);
                    }
                }
            }
            // Unmatched exterior edges impose zero normal flux.
        }
        int count = int(result.expected.size());
        MPI_Bcast(&count, 1, MPI_INT, 0, mpi.comm);
        result.expected.resize(size_t(count));
        MPI_Bcast(result.expected.data(), count, MPI_DOUBLE, 0, mpi.comm);
        reader.BuildCell2Cell();
        UnstructuredMeshSerialRW::PartitionOptions options;
        options.metisType = "KWAY";
        options.metisUfactor = 30;
        options.metisSeed = 42;
        options.metisNcuts = 1;
        reader.MeshPartitionCell2Cell(options);
        reader.PartitionReorderToMeshCell2Cell();
        mesh.RecoverNode2CellAndNode2Bnd();
        mesh.RecoverCell2CellAndBnd2Cell();
        mesh.BuildGhostPrimary();
        mesh.AdjGlobal2LocalPrimary();
        mesh.InterpolateFace();
        mesh.AssertOnFaces();
        return result;
    }

    struct Partition
    {
        std::vector<Index> core, ring;
        R::VersionStamp stamp;
    };

    inline Partition Classify(UnstructuredMesh &mesh, UnstructuredMeshView view)
    {
        Partition result;
        result.stamp = mesh.versions.Stamp(R::Change::Topology | R::Change::Distribution | R::Change::Layout);
        for (Index cell = 0; cell < view.NumCell(); ++cell)
        {
            bool core = true;
            auto faces = view.CellFaces(cell);
            for (size_t slot = 0; slot < faces.size(); ++slot)
            {
                const auto other = view.CellFaceOther(cell, faces[slot], rowsize(slot));
                DNDS_check_throw_info(other == UnInitIndex || (other >= 0 && other < view.NumCellProc()),
                                      "owned operator stencil is outside local closure");
                core &= other == UnInitIndex || other < view.NumCell();
            }
            (core ? result.core : result.ring).push_back(cell);
        }
        return result;
    }

    inline R::Array<Index> RowList(const std::vector<Index> &rows)
    {
        R::Array<Index> result(R::RowLayout::Dense(rows.size(), 1));
        auto write = result.Write();
        for (size_t i = 0; i < rows.size(); ++i)
            write.View()(i, 0) = rows[i];
        return result;
    }

    inline R::Array<real> OwnedField(UnstructuredMesh &mesh)
    {
        R::Array<real> result(R::RowLayout::Dense(size_t(mesh.NumCell()), 1));
        auto write = result.Write();
        for (Index cell = 0; cell < mesh.NumCell(); ++cell)
            write.View()(size_t(cell), 0) = ManufacturedValue(mesh.cell2cellOrig(cell, 0));
        return result;
    }

    inline real Expected(const Fixture &fixture, Index localCell)
    {
        const auto original = fixture.mesh->cell2cellOrig(localCell, 0);
        DNDS_check_throw_info(original >= 0 && size_t(original) < fixture.expected.size(), "original cell ID outside serial oracle");
        return fixture.expected[size_t(original)];
    }
}
