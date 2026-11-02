#include "Cauchy/Core/Domain.hpp"
#include "DNDS/Runtime/Array/ArrayIO.hpp"
#include "DNDS/Serializer/SerializerH5.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "DNDS/Runtime/Communication/IdRowTransfer.hpp"

namespace DNDS::Cauchy
{
    namespace
    {
        index CanonicalPeriodicZone(index zone)
        {
            if (zone == Geom::BC_ID_PERIODIC_1_DONOR)
                return Geom::BC_ID_PERIODIC_1;
            if (zone == Geom::BC_ID_PERIODIC_2_DONOR)
                return Geom::BC_ID_PERIODIC_2;
            if (zone == Geom::BC_ID_PERIODIC_3_DONOR)
                return Geom::BC_ID_PERIODIC_3;
            return zone;
        }
        Json FlowSchema(const EquationSystem &equation)
        {
            const auto &schema = equation.Schema();
            Json blocks = Json::array();
            for (const auto &block : schema.blocks)
            {
                Json components = Json::array();
                for (const auto &component : block.components)
                    components.push_back({{"name", component.name}, {"units", component.units}, {"conservation_group", component.conservationGroup}});
                blocks.push_back({{"name", block.name}, {"components", components}, {"vector_starts", block.vectorStarts}});
            }
            return {{"family", schema.family}, {"energy_convention", schema.energyConvention}, {"physical_accumulation", schema.physicalAccumulation}, {"blocks", blocks}, {"eos", "perfectGas"}, {"gamma", equation.Gas().gamma}, {"gas_constant", equation.Gas().gasConstant}};
        }
        Json CompatibleConfig(Json config)
        {
            // Time/output/placement/execution choices and polynomial setup do
            // not change stored means. Keep other physical/numerical choices
            // explicit: e.g. an old manual Roe stencil must not silently become
            // automatic. Raw non-root caller metadata is still compared exactly.
            if (!config.is_object() || !config.contains("domains") || !config["domains"].is_array())
                return config;
            config.erase("time");
            config.erase("output");
            for (auto &domain : config["domains"])
            {
                domain.erase("ranks");
                if (!domain.contains("settings") || !domain["settings"].is_object())
                    continue;
                auto &settings = domain["settings"];
                settings.erase("execution");
                settings.erase("implicit");
                if (settings.contains("mesh") && settings["mesh"].is_object())
                    settings["mesh"].erase("file");
                if (settings.contains("numerics") && settings["numerics"].is_object())
                {
                    auto &numerics = settings["numerics"];
                    numerics.erase("cfv");
                    if (numerics.contains("riemann") && numerics["riemann"].is_object() && !numerics["riemann"].contains("useStencil"))
                        numerics["riemann"]["useStencil"] = false;
                }
            }
            return config;
        }
        index StoredRows(const Serializer::SerializerBaseSSP &serializer, const std::string &name)
        {
            const auto previous = serializer->GetCurrentPath();
            serializer->GoToPath(name);
            int format = 0, ranks = 0;
            serializer->ReadInt("runtime_array_format", format);
            serializer->ReadInt("writer_ranks", ranks);
            index rows = 0;
            if (format >= 3)
                serializer->ReadIndex("rows_global", rows);
            else
            {
                serializer->GoToPath("rows");
                std::vector<index> counts;
                auto region = Serializer::ArrayGlobalOffset{ranks, 0};
                serializer->ReadIndexVector("size", counts, region);
                for (const auto count : counts)
                {
                    DNDS_check_throw_info(count >= 0 && count <= INT64_MAX - rows, "invalid restart row count");
                    rows += count;
                }
            }
            serializer->GoToPath(previous);
            DNDS_check_throw_info(rows >= 0, "negative restart row count");
            return rows;
        }
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
            std::vector<Geom::NodePeriodicBits> faceBits;
            for (index iCell = 0; iCell < view.NumCell(); ++iCell)
            {
                auto row = topology.View()[iCell];
                const auto nodes = view.CellNodes(iCell), faces = view.CellFaces(iCell);
                auto element = view.GetCellElement(iCell);
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
                    // A periodic face's main/donor zone and cell2face bits
                    // describe its selected stored image. Native topology
                    // selects that image from ordered parents; a new partition
                    // may select the complementary image. Archive physical
                    // axis identity and the cell's own face-node frame instead.
                    row[offset] = CanonicalPeriodicZone(view.GetFaceZone(iFace));
                    row[offset + 1] = other == UnInitIndex ? UnInitIndex : mesh.OriginalEntityId(Geom::EntityKind::Cell, other);
                    uint8_t frame = 0;
                    if (view.IsPeriodic())
                    {
                        faceBits.resize(element.ObtainFace(slot).GetNumNodes());
                        element.ExtractFaceNodes(slot, view.CellNodePeriodicBits(iCell), faceBits);
                        frame = 7;
                        for (const auto bit : faceBits)
                            frame &= uint8_t(bit);
                    }
                    row[offset + 2] = frame;
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
        template <class T>
        std::string RowDifference(const Runtime::Array<T> &left, const Runtime::Array<T> &right, const Runtime::Array<index> &ids)
        {
            if (left.RowCount() != right.RowCount())
                return fmt::format("row counts {} != {}", left.RowCount(), right.RowCount());
            const auto a = left.Read(), b = right.Read();
            const auto labels = ids.Read();
            for (index iRow = 0; iRow < left.RowCount(); ++iRow)
            {
                if (left.RowSize(iRow) != right.RowSize(iRow))
                    return fmt::format("cell {} widths {} != {}", labels.View()(iRow, 0), left.RowSize(iRow), right.RowSize(iRow));
                for (index j = 0; j < left.RowSize(iRow); ++j)
                    if (a.View()(iRow, j) != b.View()(iRow, j))
                        return fmt::format("cell {} slot {}: {} != {}", labels.View()(iRow, 0), j, a.View()(iRow, j), b.View()(iRow, j));
            }
            return {};
        }
        std::string TopologyDifference(const Runtime::Array<index> &left, const Runtime::Array<index> &right,
                                       const Runtime::Array<index> &ids, bool legacyFaceImage)
        {
            if (left.RowCount() != right.RowCount())
                return "topology row counts differ";
            const auto a = left.Read(), b = right.Read(), labels = ids.Read();
            for (index iRow = 0; iRow < left.RowCount(); ++iRow)
            {
                if (left.RowSize(iRow) != right.RowSize(iRow))
                    return fmt::format("cell {} topology widths differ", labels.View()(iRow, 0));
                const index faceStart = 4 + 2 * b.View()(iRow, 2);
                for (index j = 0; j < left.RowSize(iRow); ++j)
                {
                    index stored = a.View()(iRow, j), current = b.View()(iRow, j);
                    if (j >= faceStart)
                    {
                        const index field = (j - faceStart) % 3;
                        if (field == 0)
                        {
                            stored = CanonicalPeriodicZone(stored);
                            current = CanonicalPeriodicZone(current);
                        }
                        if (field == 2 && legacyFaceImage)
                        {
                            // Old archives stored a partition-dependent face
                            // image gauge, not a physical mesh invariant. Cell
                            // node IDs/bits/coordinates, neighbor IDs, periodic
                            // axis and global transformations remain exact.
                            if (stored < 0 || stored > 7)
                                return "invalid legacy periodic image bits";
                            continue;
                        }
                    }
                    if (stored != current)
                        return fmt::format("cell {} slot {}: {} != {}", labels.View()(iRow, 0), j, stored, current);
                }
            }
            return {};
        }
    }
    void Domain::WriteCheckpoint(const std::string &path, const Json &resolved, bool includeReconstruction) const
    {
        const MPIInfo mpi(_fv.Metrics().Communicator());
        const bool history = _timeStepper && _timeStepper->Method() == "bdf2" && _timeStepper->HasHistory();
        int flags[]{int(includeReconstruction), -int(includeReconstruction), int(history), -int(history)};
        MPI_Allreduce(MPI_IN_PLACE, flags, 4, MPI_INT, MPI_MIN, mpi.comm);
        DNDS_check_throw_info(flags[0] == -flags[1] && flags[2] == -flags[3], "checkpoint optional sections differ between ranks");
        const Json metadata{{"schema_version", 1}, {"mesh_envelope", 2}, {"flow", FlowSchema(_equation)}, {"reconstruction", _fv.ReconstructionMetadata()}, {"time_method", _timeStepper ? _timeStepper->Method() : "ssprk3"}, {"history", history}};
        auto serializer = std::make_shared<Serializer::SerializerH5>(mpi);
        serializer->OpenFile(path, false);
        serializer->WriteInt("cauchy_format", 4);
        serializer->WriteString("resolved_config", resolved.dump());
        serializer->WriteString("metadata_json", metadata.dump());
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
        if (includeReconstruction)
            Runtime::WriteArray(_restoreReconstruction ? *_reconstructionBackup : _fv.ReconstructionGuess(*_workspace).Owned(), serializer, "reconstruction_guess", mpi);
        if (history)
        {
            serializer->WriteReal("history_dt", _timeStepper->PreviousDt());
            Runtime::WriteArray(_timeStepper->PreviousState().Owned(), serializer, "history_flow", mpi);
        }
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
        DNDS_check_throw_info(version >= 2 && version <= 4, "unsupported Cauchy checkpoint format");
        const auto rowCount = StoredRows(serializer, "cell_ids");
        const auto range = EvenSplitRange(mpi.rank, mpi.size, rowCount);
        const Serializer::ArrayGlobalOffset shard{range.second - range.first, range.first};
        auto ids = Runtime::ReadArray<index, Runtime::FixedRows<1>>(serializer, "cell_ids", mpi, shard);
        auto topology = Runtime::ReadArray<index>(serializer, "mesh_topology", mpi, shard);
        auto coordinates = Runtime::ReadArray<real>(serializer, "mesh_coordinates", mpi, shard);
        auto transforms = Runtime::ReadArray<real>(serializer, "periodic_transforms", mpi, {1, 0});
        auto flow = Runtime::ReadArray<real, Runtime::FixedRows<5>>(serializer, "flow", mpi, shard);
        Json metadata;
        if (version == 4)
        {
            std::string serialized;
            serializer->ReadString("metadata_json", serialized);
            metadata = Json::parse(serialized, nullptr, false);
        }
        std::optional<Runtime::Array<real>> reconstruction;
        const auto entries = serializer->ListCurrentPath();
        if (entries.count("reconstruction_guess"))
            reconstruction.emplace(Runtime::ReadArray<real>(serializer, "reconstruction_guess", mpi, shard));
        std::optional<Runtime::Array<real>> history;
        real historyDt = 0;
        if (version == 4 && metadata.is_object() && metadata.value("history", false))
        {
            serializer->ReadReal("history_dt", historyDt);
            history.emplace(Runtime::ReadArray<real, Runtime::FixedRows<5>>(serializer, "history_flow", mpi, shard));
        }
        serializer->CloseFile();
        auto storedConfig = Json::parse(config, nullptr, false);
        int valid = name == _name && !storedConfig.is_discarded() && CompatibleConfig(storedConfig) == CompatibleConfig(resolved) && std::isfinite(time) && time >= 0 && step >= 0 &&
                    flow.RowCount() == ids.RowCount() && flow.Layout().Width() == 5;
        valid = valid && (version < 4 || (metadata.is_object() && metadata.value("schema_version", 0) == 1 && metadata.value("flow", Json()) == FlowSchema(_equation)));
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, mpi.comm);
        DNDS_check_throw_info(valid, "Cauchy restart config/equation schema mismatch");
        const int meshEnvelope = version == 4 ? metadata.value("mesh_envelope", 1) : 1;
        DNDS_check_throw_info(meshEnvelope == 1 || meshEnvelope == 2, "unsupported Cauchy mesh envelope");
        auto context = std::make_shared<Runtime::CommunicationContext>(mpi.comm);
        const auto targetIds = CellIds(*this);
        const auto route = Runtime::BuildRowTransferById(context, ids, targetIds, true);
        const auto transfer = [&](const auto &source, auto &destination)
        {
            using T = typename std::decay_t<decltype(source)>::value_type;
            Runtime::PayloadPlan(route, source.Layout(), destination.Layout(), Runtime::ElementSchema::For<T>())
                .Bind(source, destination, Runtime::TransferTransport::Direct, {Runtime::TransferMode::Transient, context, 0})
                .Begin()
                .Wait();
        };
        std::optional<MeshEnvelope> envelope;
        std::optional<Runtime::Array<index>> mappedTopology;
        std::optional<Runtime::Array<real>> mappedCoordinates;
        MPI::LocalStage(mpi, [&]
                        { envelope.emplace(DescribeMesh(*this)); });
        const auto topologyLayout = route.DestinationLayout(topology.Layout());
        const auto coordinateLayout = route.DestinationLayout(coordinates.Layout());
        MPI::LocalStage(mpi, [&]
                        {
            mappedTopology.emplace(topologyLayout);
            mappedCoordinates.emplace(coordinateLayout); });
        transfer(topology, *mappedTopology);
        transfer(coordinates, *mappedCoordinates);
        MPI::LocalStage(mpi, [&]
                        {
            const auto topologyDifference = TopologyDifference(*mappedTopology, envelope->topology, targetIds, meshEnvelope == 1);
            const auto coordinateDifference = RowDifference(*mappedCoordinates, envelope->coordinates, targetIds);
            DNDS_check_throw_info(topologyDifference.empty() && coordinateDifference.empty() && SameRows(transforms, envelope->periodicTransforms),
                                  "Cauchy restart persistent-cell mesh topology/geometry mismatch: topology " + topologyDifference + "; coordinates " + coordinateDifference); });
        _trialReady = false;
        // Existing trial allocations and binding identities remain stable.
        auto &trial = _trial.Block("flow");
        transfer(flow, trial.Owned());
        DNDS_check_throw_info(_fv.Admissible(trial), "Cauchy restart contains inadmissible states");
        Runtime::Array<real> guess;
        MPI::LocalStage(mpi, [&]
                        { guess = Runtime::Array<real>(_fv.ReconstructionGuess(*_workspace).Owned().Layout()); });
        int compatibleReconstruction = bool(reconstruction) && (version < 4 || metadata.value("reconstruction", Json()) == _fv.ReconstructionMetadata());
        if (reconstruction)
            for (index iRow = 0; iRow < reconstruction->RowCount(); ++iRow)
                compatibleReconstruction = compatibleReconstruction && reconstruction->RowSize(iRow) == guess.Layout().Width();
        MPI_Allreduce(MPI_IN_PLACE, &compatibleReconstruction, 1, MPI_INT, MPI_MIN, mpi.comm);
        if (compatibleReconstruction)
            transfer(*reconstruction, guess);
        else
        {
            auto write = guess.Write();
            const auto output = write.View<Runtime::UniformTableRows>();
            for (index iCell = 0; iCell < output.RowCount(); ++iCell)
                for (index j = 0; j < output.RowSize(iCell); ++j)
                    output(iCell, j) = 0;
        }
        StateEpoch epoch;
        epoch.acceptedStep = uint64_t(step);
        std::unique_ptr<ODE::FieldTimeStepper> restoredStepper;
        std::optional<CFV::FVField> restoredHistory;
        if (_timeStepper)
        {
            MPI::LocalStage(mpi, [&]
                            { restoredStepper = _timeStepper->Fresh(); });
            if (history && metadata.value("time_method", std::string{}) == restoredStepper->Method() && restoredStepper->Method() == "bdf2")
            {
                MPI::LocalStage(mpi, [&]
                                {
                    DNDS_check_throw_info(std::isfinite(historyDt) && historyDt > 0, "invalid Cauchy restart history timestep");
                    restoredHistory.emplace(_fv.Metrics().BuildField(Flow().Owned().Layout())); });
                transfer(*history, restoredHistory->Owned());
                DNDS_check_throw_info(_fv.Admissible(*restoredHistory), "inadmissible Cauchy restart history");
                MPI::LocalStage(mpi, [&]
                                { restoredStepper->RestoreHistory(*restoredHistory, historyDt); });
            }
        }
        PublishTrial(time, epoch, &guess);
        if (restoredStepper)
            _timeStepper.swap(restoredStepper);
        _nextTrial = 0;
    }
}
