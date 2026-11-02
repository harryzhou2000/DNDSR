#include "Cauchy/Core/Domain.hpp"
#include "DNDS/Runtime/Array/ArrayIO.hpp"
#include "DNDS/Serializer/SerializerH5.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::Cauchy
{
    namespace
    {
        Runtime::Array<index> CellIds(const Domain &domain)
        {
            const auto &cells = domain.Discretization().Metrics().Cells();
            Runtime::Array<index> ids(Runtime::RowLayout::Dense(cells.NumOwned(), 1));
            auto write = ids.Write();
            for (index iCell = 0; iCell < cells.NumOwned(); ++iCell)
                write.View()(iCell, 0) = domain.Discretization().Mesh().OriginalEntityId(Geom::EntityKind::Cell, iCell);
            return ids;
        }
        struct MeshEnvelope
        {
            Runtime::Array<index> topology;
            Runtime::Array<real> coordinates;
            Runtime::Array<real> periodicTransforms;
        };
        // Exact per-owned-cell records, not a collision-prone hash. These are
        // restart metadata; the numerical flow payload is never packed/copied.
        MeshEnvelope DescribeMesh(const Domain &domain)
        {
            const auto &mesh = domain.Discretization().Mesh();
            // The metric read pins the same facade publication and its CellFlux
            // groups. Borrow only while this scope prevents mesh replacement.
            auto read = domain.Discretization().Metrics().AcquireRead();
            const auto state = mesh.CurrentNativeState();
            const auto view = state->BorrowHostView();
            std::vector<size_t> topologySizes, coordinateSizes;
            for (index iCell = 0; iCell < view.NumCell(); ++iCell)
            {
                topologySizes.push_back(4 + 2 * view.CellNodes(iCell).size() + 3 * view.CellFaces(iCell).size());
                coordinateSizes.push_back(3 * view.CellNodes(iCell).size());
            }
            MeshEnvelope envelope{Runtime::Array<index>(Runtime::RowLayout::Ragged(topologySizes)),
                                  Runtime::Array<real>(Runtime::RowLayout::Ragged(coordinateSizes)),
                                  Runtime::Array<real>(Runtime::RowLayout::Dense(1, view.IsPeriodic() ? 45 : 0))};
            if (view.IsPeriodic())
            {
                auto transforms = envelope.periodicTransforms.Write();
                const auto periodic = view.Periodic();
                for (int axis = 1; axis <= 3; ++axis)
                    for (int d = 0; d < 3; ++d)
                    {
                        const int base = (axis - 1) * 15;
                        transforms.View()(0, base + d) = periodic.translation[axis].map()[d];
                        transforms.View()(0, base + 3 + d) = periodic.rotationCenter[axis].map()[d];
                        for (int k = 0; k < 3; ++k)
                            transforms.View()(0, base + 6 + 3 * d + k) = periodic.rotation[axis].map()(d, k);
                    }
            }
            auto topology = envelope.topology.Write();
            auto coordinates = envelope.coordinates.Write();
            for (index iCell = 0; iCell < view.NumCell(); ++iCell)
            {
                auto row = topology.View()[iCell];
                const auto nodes = view.CellNodes(iCell), faces = view.CellFaces(iCell);
                row[0] = index(view.GetCellInfo(iCell).getElemType());
                row[1] = view.GetCellZone(iCell);
                row[2] = nodes.size();
                row[3] = faces.size();
                for (rowsize slot = 0; slot < rowsize(nodes.size()); ++slot)
                {
                    row[4 + 2 * slot] = mesh.OriginalEntityId(Geom::EntityKind::Node, nodes[slot]);
                    row[5 + 2 * slot] = view.IsPeriodic() ? index(uint8_t(view.CellNodePeriodicBits(iCell)[slot])) : 0;
                    const auto point = view.GetCoordNodeOnCell(iCell, slot);
                    for (int d = 0; d < 3; ++d)
                        coordinates.View()(iCell, 3 * slot + d) = point[d];
                }
                for (rowsize slot = 0; slot < rowsize(faces.size()); ++slot)
                {
                    const auto iFace = faces[slot], other = view.CellFaceOther(iCell, iFace, slot);
                    const index offset = 4 + 2 * nodes.size() + 3 * slot;
                    row[offset] = view.GetFaceZone(iFace);
                    row[offset + 1] = other == UnInitIndex ? UnInitIndex : mesh.OriginalEntityId(Geom::EntityKind::Cell, other);
                    row[offset + 2] = view.IsPeriodic() ? index(uint8_t(view.CellFacePeriodicBits(iCell)[slot])) : 0;
                }
            }
            return envelope;
        }
        template <class T>
        bool SameRows(const Runtime::Array<T> &left, const Runtime::Array<T> &right)
        {
            if (left.RowCount() != right.RowCount())
                return false;
            auto a = left.Read(), b = right.Read();
            for (index iRow = 0; iRow < left.RowCount(); ++iRow)
            {
                if (left.RowSize(iRow) != right.RowSize(iRow))
                    return false;
                for (index j = 0; j < left.RowSize(iRow); ++j)
                    if (a.View()(iRow, j) != b.View()(iRow, j))
                        return false;
            }
            return true;
        }
    }
    void Domain::WriteCheckpoint(const std::string &path, const Json &resolved) const
    {
        const MPIInfo mpi(_fv.Metrics().Communicator());
        auto serializer = std::make_shared<Serializer::SerializerH5>(mpi);
        serializer->OpenFile(path, false);
        serializer->WriteInt("cauchy_format", 3);
        serializer->WriteString("resolved_config", resolved.dump());
        serializer->WriteString("domain_name", _name);
        serializer->WriteReal("accepted_time", _time);
        serializer->WriteIndex("accepted_step", index(_epoch.acceptedStep));
        Runtime::WriteArray(CellIds(*this), serializer, "cell_ids", mpi);
        std::optional<MeshEnvelope> envelope;
        MPI::LocalStage(mpi, [&]
                        { envelope.emplace(DescribeMesh(*this)); });
        Runtime::WriteArray(envelope->topology, serializer, "mesh_topology", mpi);
        Runtime::WriteArray(envelope->coordinates, serializer, "mesh_coordinates", mpi);
        Runtime::WriteArray(envelope->periodicTransforms, serializer, "periodic_transforms", mpi);
        Runtime::WriteArray(Flow().Owned(), serializer, "flow", mpi);
        Runtime::WriteArray(_fv.ReconstructionGuess(*_workspace).Owned(), serializer, "reconstruction_guess", mpi);
        serializer->CloseFile();
    }
    void Domain::ReadCheckpoint(const std::string &path, const Json &resolved)
    {
        const MPIInfo mpi(_fv.Metrics().Communicator());
        auto serializer = std::make_shared<Serializer::SerializerH5>(mpi);
        serializer->OpenFile(path, true);
        int version = 0;
        std::string config, name;
        real time = 0;
        index step = 0;
        serializer->ReadInt("cauchy_format", version);
        serializer->ReadString("resolved_config", config);
        serializer->ReadString("domain_name", name);
        serializer->ReadReal("accepted_time", time);
        serializer->ReadIndex("accepted_step", step);
        auto ids = Runtime::ReadArray<index>(serializer, "cell_ids", mpi);
        auto topology = Runtime::ReadArray<index>(serializer, "mesh_topology", mpi);
        auto coordinates = Runtime::ReadArray<real>(serializer, "mesh_coordinates", mpi);
        auto transforms = Runtime::ReadArray<real>(serializer, "periodic_transforms", mpi);
        auto flow = Runtime::ReadArray<real>(serializer, "flow", mpi);
        std::optional<Runtime::Array<real>> reconstruction;
        if (version == 3)
            reconstruction.emplace(Runtime::ReadArray<real>(serializer, "reconstruction_guess", mpi));
        serializer->CloseFile();
        auto storedConfig = Json::parse(config, nullptr, false);
        // Earlier Cauchy archives supplied only manual Riemann correction
        // scalars. Preserve that numerical behavior when the new option is
        // absent; loading under automatic stencils still rejects explicitly.
        if (storedConfig.is_object() && storedConfig.contains("domains") && storedConfig["domains"].is_array())
            for (auto &domain : storedConfig["domains"])
            {
                const Json::json_pointer pointer("/settings/numerics/riemann");
                if (domain.contains(pointer) && domain.at(pointer).is_object() && !domain.at(pointer).contains("useStencil"))
                    domain.at(pointer)["useStencil"] = false;
            }
        int valid = (version == 2 || version == 3) && name == _name && storedConfig == resolved && std::isfinite(time) && time >= 0 && step >= 0 &&
                    ids.RowCount() == _fv.Metrics().NumCell() && ids.Layout().Width() == 1 && flow.RowCount() == _fv.Metrics().NumCell() && flow.Layout().Width() == 5;
        MPI::LocalStage(mpi, [&]
                        {
        if (valid) {
            const auto envelope = DescribeMesh(*this);
            valid = SameRows(ids, CellIds(*this)) && SameRows(topology, envelope.topology) && SameRows(coordinates, envelope.coordinates) &&
                    SameRows(transforms, envelope.periodicTransforms);
        } });
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, mpi.comm);
        DNDS_check_throw_info(valid, "Cauchy restart config/schema/partition/persistent IDs/mesh mismatch");
        _trialReady = false;
        // Existing trial allocations and binding identities remain stable.
        auto &trial = _trial.Block("flow");
        {
            auto read = flow.Read();
            auto write = trial.Owned().Write();
            for (index iCell = 0; iCell < _fv.Metrics().NumCell(); ++iCell)
                for (int j = 0; j < 5; ++j)
                    write.View()(iCell, j) = read.View()(iCell, j);
        }
        DNDS_check_throw_info(_fv.Admissible(trial), "Cauchy restart contains inadmissible states");
        if (!reconstruction)
        {
            // Format2 predates persistent P3 warm starts. Its means remain
            // readable, but finite-sweep continuation uses a cold initial guess.
            reconstruction.emplace(_fv.ReconstructionGuess(*_workspace).Owned().Layout());
            auto write = reconstruction->Write();
            const auto output = write.View<Runtime::UniformTableRows>();
            for (index iCell = 0; iCell < output.RowCount(); ++iCell)
                for (index j = 0; j < output.RowSize(iCell); ++j)
                    output(iCell, j) = 0;
        }
        StateEpoch epoch;
        epoch.acceptedStep = uint64_t(step);
        PublishTrial(time, epoch, &*reconstruction);
        _nextTrial = 0;
    }
}
