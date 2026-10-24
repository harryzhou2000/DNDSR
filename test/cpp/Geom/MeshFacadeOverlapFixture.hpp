#pragma once

#include "Geom/Mesh/Mesh.hpp"
#include "Geom/Mesh/EntitySpace.hpp"
#include "Geom/Mesh/MeshTransition.hpp"
#include "Geom/Mesh/FieldTransfer.hpp"
#include "DNDS/Runtime/EagerArray.hpp"
#include "DNDS/Runtime/MPIProgress.hpp"
#include "DNDS/Runtime/ArrayIO.hpp"
#include "DNDS/Serializer/SerializerH5.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <map>
#include <vector>
#include <unistd.h>

namespace DNDS::Geom::OverlapTest
{
    using Index = DNDS::index;
    namespace R = DNDS::Runtime;

    struct Fixture
    {
        ssp<UnstructuredMesh> mesh;
        std::vector<real> expected;
        std::optional<R::Array<real>> restartedOwned;
        std::optional<EntitySpace> restartedOrder;
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
            write.View()(size_t(cell), 0) = ManufacturedValue(mesh.OriginalEntityId(EntityKind::Cell, cell));
        return result;
    }

    inline R::Array<real> OwnedField(Fixture &fixture)
    {
        if (!fixture.restartedOwned)
            return OwnedField(*fixture.mesh);
        DNDS_check_throw_info(fixture.restartedOrder && fixture.restartedOrder->SameOrder(fixture.mesh->Entities(EntityKind::Cell)),
                              "restarted solution no longer matches current cell order");
        return *fixture.restartedOwned;
    }

    inline real Expected(const Fixture &fixture, Index localCell)
    {
        const auto original = fixture.mesh->OriginalEntityId(EntityKind::Cell, localCell);
        DNDS_check_throw_info(original >= 0 && size_t(original) < fixture.expected.size(), "original cell ID outside serial oracle");
        return fixture.expected[size_t(original)];
    }

    /// Exercise independent native ownership and a real distributed arrival,
    /// retaining the old source throughout preparation/publication.
    inline void RedistributeCurrent(Fixture &fixture)
    {
        auto source = fixture.mesh;
        auto retained = source->AcquireRead(MeshReadRequirements::CellFlux());
        auto sourceOrder = source->Entities(EntityKind::Cell);
        MeshTransitionHandle handle(source);
        ReorderInput input;
        input.explicitMaps.push_back({EntityKind::Cell,
                                      std::vector<MPI_int>(size_t(source->NumCell()), (source->mpi.rank + 1) % source->mpi.size)});
        auto prepared = PreparedMeshTransition::Prepare(handle, input);
        prepared.Publish();
        fixture.mesh = handle.Get();
        DNDS_check_throw_info(sourceOrder.Valid() && !sourceOrder.SameDomain(fixture.mesh->Entities(EntityKind::Cell)),
                              "detached publication must retain the old source and distinguish its order");
        if (!fixture.mesh->HasNativeState())
            fixture.mesh->PublishNative(MeshReadRequirements::CellFlux());
    }
    /// Checkpoint the actual solution and stable cell IDs, then execute the
    /// same stencil on restored values rather than regenerating its input.
    inline void RestartCurrent(Fixture &fixture)
    {
        auto source = fixture.mesh;
        const auto mpi = source->mpi;
        auto retained = source->AcquireRead(MeshReadRequirements::CellFlux());
        const auto sourceOrder = source->Entities(EntityKind::Cell);
        auto field = OwnedField(fixture);
        const auto fieldIdentity = field.Identity(), fieldVersion = field.Version();
        R::Array<Index> ids(R::RowLayout::Dense(size_t(source->NumCell()), 1));
        {
            auto write = ids.Write();
            for (Index row = 0; row < source->NumCell(); ++row)
                write.View()(size_t(row), 0) = source->OriginalEntityId(EntityKind::Cell, row);
        }
        int rootPid = int(getpid());
        MPI_Bcast(&rootPid, 1, MPI_INT, 0, mpi.comm);
        const auto filename = (std::filesystem::temp_directory_path() /
                               ("dnds-facade-overlap-restart-" + std::to_string(rootPid) + "-np" + std::to_string(mpi.size) + ".h5"))
                                  .string();
        constexpr const char *fieldSchema = "Geom.Overlap.Solution.f64.v1";
        constexpr const char *idSchema = "Geom.Overlap.CellOriginal.i64.v1";
        auto serializer = make_ssp<Serializer::SerializerH5>(mpi);
        serializer->SetCollectiveRW(true, true);
        {
            auto fieldRead = field.Read();
            auto idRead = ids.Read();
            serializer->OpenFile(filename, false);
            WriteCurrentPrimaryMeshSnapshot(*source, serializer, "mesh");
            R::WriteArray(field, serializer, "solution", mpi, fieldSchema);
            R::WriteArray(ids, serializer, "original_cells", mpi, idSchema);
            serializer->CloseFile();
        }
        auto restored = make_ssp<UnstructuredMesh>(mpi, 2);
        serializer->OpenFile(filename, true);
        restored->ReadSerialize(serializer, "mesh");
        auto loaded = R::ReadArray<real>(serializer, "solution", mpi, Serializer::ArrayGlobalOffset_Unknown,
                                         R::Device::Host(), fieldSchema);
        auto loadedIds = R::ReadArray<Index>(serializer, "original_cells", mpi, Serializer::ArrayGlobalOffset_Unknown,
                                             R::Device::Host(), idSchema);
        serializer->CloseFile();
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            DNDS_check_throw_info(loaded.Rows() == size_t(restored->NumCell()) && loadedIds.Rows() == loaded.Rows(),
                                  "restart solution/identity extents differ from mesh");
            auto values = loaded.Read();
            auto identity = loadedIds.Read();
            for (Index row = 0; row < restored->NumCell(); ++row)
            {
                const auto original = restored->OriginalEntityId(EntityKind::Cell, row);
                DNDS_check_throw_info(identity.View()(size_t(row), 0) == original &&
                                      values.View()(size_t(row), 0) == ManufacturedValue(original),
                                      "restart solution/mesh stable-ID oracle differs");
            }
            DNDS_check_throw_info(field.Identity() == fieldIdentity && field.Version() == fieldVersion &&
                                  loaded.Identity() != fieldIdentity && sourceOrder.Valid(),
                                  "checkpoint changed source field or failed to allocate independent restart values"); });
        restored->RecoverNode2CellAndNode2Bnd();
        restored->RecoverCell2CellAndBnd2Cell();
        restored->BuildGhostPrimary();
        restored->AdjGlobal2LocalPrimary();
        restored->InterpolateFace();
        restored->AssertOnFaces();
        restored->PublishNative(MeshReadRequirements::CellFlux());
        fixture.mesh = std::move(restored);
        fixture.restartedOwned.emplace(std::move(loaded));
        fixture.restartedOrder.emplace(fixture.mesh->Entities(EntityKind::Cell));
        DNDS_check_throw_info(!sourceOrder.SameDomain(*fixture.restartedOrder) && retained.HostView().NumCell() == sourceOrder.NumOwned(),
                              "restart must preserve the old read while publishing an independent mesh");
        MPI_Barrier(mpi.comm);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            if (mpi.rank == 0)
            {
                std::error_code error;
                std::filesystem::remove(filename, error);
                DNDS_check_throw_info(!error, "cannot remove overlap restart fixture");
            } });
        MPI_Barrier(mpi.comm);
    }

}
