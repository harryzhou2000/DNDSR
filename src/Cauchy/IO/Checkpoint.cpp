#include "DNDS/MPI/Failure.hpp"
#include "Cauchy/Core/Domain.hpp"
#include "Cauchy/IO/StateArchive.hpp"
#include "DNDS/Runtime/Array/ArrayIO.hpp"
#include "DNDS/Serializer/SerializerH5.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "DNDS/Runtime/Communication/IdRowTransfer.hpp"
#include <limits>
#include <set>

namespace DNDS::Cauchy
{
    namespace
    {
        void RemoveInactiveQCR(Json &model)
        {
            // This optional switch did not exist in early RANS checkpoints.
            // Only the inactive default normalizes away; true remains physical.
            if (model.is_object() && !model.value("useQCR", false))
                model.erase("useQCR");
        }
        nlohmann::json CompatibleEquationMetadata(Json value)
        {
            if (value.is_object() && value.contains("rans"))
                RemoveInactiveQCR(value["rans"]);
            return value;
        }
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
        Json EquationMetadata(const EquationSystem &equation)
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
            Json result{{"family", schema.family}, {"energy_convention", schema.energyConvention}, {"physical_accumulation", schema.physicalAccumulation}, {"blocks", blocks}};
            if (equation.HasMixture())
            {
                const auto &material = equation.Mixture();
                const auto &scales = material.Scales();
                result["eos"] = "idealGasMixture";
                result["species"] = material.SpeciesNames();
                result["reference_scales"] = {scales.rho0, scales.U0, scales.L0, scales.T0};
                result["base_temperature"] = material.BaseTemperature();
                result["base_internal_energies"] = material.BaseSpecificInternalEnergies();
                result["model_description"] = material.ModelDescription();
            }
            else
            {
                result["eos"] = "perfectGas";
                result["gamma"] = equation.Gas().gamma;
                result["gas_constant"] = equation.Gas().gasConstant;
            }
            if (equation.HasRANS())
                result["rans"] = equation.RANS().Settings();
            return result;
        }
        nlohmann::json CompatibleConfig(Json config)
        {
            // Display/config JSON preserves insertion order. Semantic restart
            // equality must not: filling old defaults appends keys at another
            // position. Return ordinary object JSON for order-independent keys.
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
                if (settings.value("initialTemperature", real(0)) == 0)
                    settings.erase("initialTemperature");
                if (settings.contains("initialMassFractions") && settings["initialMassFractions"].is_object() && settings["initialMassFractions"].empty())
                    settings.erase("initialMassFractions");
                if (settings.contains("equation") && settings["equation"].value("eos", std::string("perfectGas")) == "perfectGas")
                    settings["equation"].erase("mixture");
                if (settings.contains("equation") && settings["equation"].contains("rans"))
                    RemoveInactiveQCR(settings["equation"]["rans"]);
                settings.erase("execution");
                settings.erase("implicit");
                if (settings.contains("motion") && settings["motion"].value("angularVelocity", Vector3{}) == Vector3{} &&
                    settings["motion"].value("translationVelocity", Vector3{}) == Vector3{})
                    settings.erase("motion"); // Old fixed-frame archives had no motion section.
                if (settings.contains("boundaries") && settings["boundaries"].is_object())
                    for (auto &condition : settings["boundaries"])
                    {
                        if (condition.value("wallVelocityFrame", std::string("mesh")) == "mesh")
                            condition.erase("wallVelocityFrame");
                        // Old flow5 archives predate species boundary input.
                        // An inactive default is compatible; prescribed
                        // nonempty composition remains a physical difference.
                        if (condition.contains("massFractions") && condition["massFractions"].is_object() && condition["massFractions"].empty())
                            condition.erase("massFractions");
                    }
                if (settings.contains("mesh") && settings["mesh"].is_object())
                    settings["mesh"].erase("file");
                if (settings.contains("numerics") && settings["numerics"].is_object())
                {
                    auto &numerics = settings["numerics"];
                    // Earlier archives predate the coarse convection switch.
                    // Its false default preserves their operator; true remains
                    // a deliberate numerical compatibility difference.
                    if (!numerics.value("firstOrderConvection", false))
                        numerics.erase("firstOrderConvection");
                    // Adding inactive limiter controls must not invalidate
                    // older non-LVR checkpoints. Active LVR controls remain a
                    // numerical compatibility requirement.
                    if (numerics.value("limiter", "none") != "LVR")
                        numerics.erase("limitedVR");
                    numerics.erase("cfv");
                    // Retuning a reconstruction solve changes neither stored
                    // means nor the compatible polynomial coefficient schema.
                    numerics.erase("reconstructionIterations");
                    numerics.erase("reconstructionTolerance");
                    // Older archives predate this opt-in Euler wall mode.
                    // An inactive option has no numerical compatibility effect.
                    if (!numerics.value("noRsOnWall", false))
                        numerics.erase("noRsOnWall");
                    // Missing viscous controls in older archives denote the
                    // original primitive-cell-gradient scheme. Selecting the
                    // polynomial operator is a visible numerical method change.
                    if (settings.contains("equation") && settings["equation"].value("transport", "inviscid") == "NewtonianFourier")
                    {
                        if (!numerics.contains("viscousGradient"))
                            numerics["viscousGradient"] = "primitive";
                        if (numerics["viscousGradient"] == "polynomial")
                        {
                            if (!numerics.contains("viscousLimited"))
                                numerics["viscousLimited"] = true;
                            if (!numerics.contains("noGRPOnWall"))
                                numerics["noGRPOnWall"] = false;
                        }
                        else
                        {
                            numerics.erase("viscousLimited");
                            numerics.erase("noGRPOnWall");
                        }
                    }
                    else
                        for (const auto *name : {"viscousGradient", "viscousLimited", "noGRPOnWall"})
                            numerics.erase(name);
                    // Earlier Cauchy archives used pointwise segment limiting.
                    // Missing controls must not silently select Euler's newer
                    // common-beta policy. Inactive PP choices do not affect a
                    // non-PP continuation and need not invalidate old archives.
                    if (numerics.value("positivity", false))
                    {
                        if (!numerics.contains("positivityReconstruction"))
                            numerics["positivityReconstruction"] = "Segment";
                        if (!numerics.contains("positivityVolumePoints"))
                            numerics["positivityVolumePoints"] = true;
                        if (numerics["positivityReconstruction"] == "Segment")
                        {
                            numerics.erase("positivityCompressToMean");
                            numerics.erase("positivityBetaPower");
                        }
                        else
                        {
                            if (!numerics.contains("positivityCompressToMean"))
                                numerics["positivityCompressToMean"] = true;
                            if (!numerics.contains("positivityBetaPower"))
                                numerics["positivityBetaPower"] = 1;
                        }
                    }
                    else
                        for (const auto *name : {"positivityReconstruction", "positivityVolumePoints", "positivityCompressToMean", "positivityBetaPower"})
                            numerics.erase(name);
                    if (numerics.contains("riemann") && numerics["riemann"].is_object() && !numerics["riemann"].contains("useStencil"))
                        numerics["riemann"]["useStencil"] = false;
                }
            }
            return config;
        }
        Json FrameMetadata(const FiniteVolume &fv, real time)
        {
            const auto frame = fv.Motion().At(time);
            std::array<real, 9> rotation;
            std::array<real, 3> translation;
            std::copy_n(frame.pose.rotation.data(), 9, rotation.data());
            std::copy_n(frame.pose.translation.data(), 3, translation.data());
            return {{"convention", "absolute-velocity-and-energy-in-mesh-axes"}, {"law", fv.Motion().Settings()}, {"rotation_column_major", rotation}, {"translation", translation}};
        }
        bool CompatibleFrame(const Json &metadata, const FiniteVolume &fv, real time)
        {
            if (!metadata.is_object() || !metadata.contains("frame"))
                return !fv.Motion().At(0).moving;
            try
            {
                const auto expected = FrameMetadata(fv, time);
                const auto &stored = metadata["frame"];
                if (stored.at("convention") != expected["convention"] || stored.at("law") != expected["law"])
                    return false;
                for (const auto *name : {"rotation_column_major", "translation"})
                {
                    const auto &a = stored.at(name), &b = expected.at(name);
                    if (!a.is_array() || a.size() != b.size())
                        return false;
                    for (index j = 0; j < index(a.size()); ++j)
                    {
                        const real x = a[j].get<real>(), y = b[j].get<real>();
                        if (!std::isfinite(x) || std::abs(x - y) > 32 * std::numeric_limits<real>::epsilon() * std::max(real(1), std::abs(y)))
                            return false;
                    }
                }
                return true;
            }
            catch (const Json::exception &)
            {
                MPI::AbortIfSolverFailure("Cauchy checkpoint validation failed");
                return false;
            }
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
    void Domain::WriteCheckpoint(const std::string &path, const Json &resolved, bool includeReconstruction,
                                 const Json &outputController) const
    {
        const MPIInfo mpi(_fv.Metrics().Communicator());
        includeReconstruction = includeReconstruction && bool(_fv.PolynomialReconstruction());
        const bool history = _timeStepper && _timeStepper->Method() == "bdf2" && _timeStepper->HasHistory();
        const bool limitedVRHistory = includeReconstruction && bool(_fv.LimitedVRHistory(*_reconstruction));
        int flags[]{int(includeReconstruction), -int(includeReconstruction), int(history), -int(history), int(limitedVRHistory), -int(limitedVRHistory)};
        MPI_Allreduce(MPI_IN_PLACE, flags, 6, MPI_INT, MPI_MIN, mpi.comm);
        DNDS_check_throw_info(flags[0] == -flags[1] && flags[2] == -flags[3] && flags[4] == -flags[5], "checkpoint optional sections differ between ranks");
        Json metadata{{"schema_version", 1}, {"mesh_envelope", 2}, {"flow", EquationMetadata(_equation)}, {"reconstruction", _fv.ReconstructionMetadata()}, {"time_method", _timeStepper ? _timeStepper->Method() : "ssprk3"}, {"history", history}, {"limited_vr_history", limitedVRHistory}, {"limited_vr_history_schema", 1}, {"frame", FrameMetadata(_fv, _time)}};
        metadata["restart_scope"] = "accepted_step";
        metadata["nonlinear_state"] = _timeStepper && _timeStepper->Method() == "ditr"
                                          ? _coupledNonlinearState.ToJson()
                                          : (_implicit ? _implicit->StateJson() : NonlinearSolveState{}.ToJson());
        metadata["state_storage"] = DescribeEquationStorage(_accepted);
        if (includeReconstruction)
            metadata["reconstruction_storage"] = DescribeEquationStorage(_fv.ReconstructionGuess(*_reconstruction));
        metadata["source_integration"] = _sourceStrang ? "strang" : "coupled";
        auto serializer = std::make_shared<Serializer::SerializerH5>(mpi);
        serializer->OpenFile(path, false);
        serializer->WriteInt("cauchy_format", 5);
        serializer->WriteString("resolved_config", resolved.dump());
        serializer->WriteString("metadata_json", metadata.dump());
        if (!outputController.is_null())
            serializer->WriteString("output_controller_json", outputController.dump());
        serializer->WriteString("domain_name", _name);
        serializer->WriteReal("accepted_time", _time);
        serializer->WriteReal("last_accepted_dt", _lastAcceptedDt);
        serializer->WriteIndex("accepted_step", index(_epoch.acceptedStep));
        Runtime::WriteArray(CellIds(*this), serializer, "cell_ids", mpi);
        std::optional<MeshEnvelope> envelope;
        MPI::LocalStage(mpi, [&]
                        { envelope.emplace(DescribeMesh(*this)); });
        Runtime::WriteArray(envelope->topology, serializer, "mesh_topology", mpi);
        Runtime::WriteArray(envelope->coordinates, serializer, "mesh_coordinates", mpi);
        Runtime::WriteArray(envelope->periodicTransforms, serializer, "periodic_transforms", mpi);
        WriteEquationArrays(_accepted, serializer, "state_field_", mpi);
        if (includeReconstruction)
            WriteEquationArrays(_restoreReconstruction ? *_reconstructionBackup : _fv.ReconstructionGuess(*_reconstruction), serializer, "reconstruction_field_", mpi);
        if (limitedVRHistory)
            Runtime::WriteArray(_restoreReconstruction ? *_limitedVRBackup : *_fv.LimitedVRHistory(*_reconstruction), serializer, "limited_vr_history", mpi);
        if (history)
        {
            serializer->WriteReal("history_dt", _timeStepper->PreviousDt());
            WriteEquationArrays(_timeStepper->PreviousState(), serializer, "history_field_", mpi);
        }
        serializer->CloseFile();
    }
    void Domain::WriteInternalCheckpoint(const std::string &path, const Json &resolved,
                                         const NonlinearSolveState &state,
                                         const Json &outputController) const
    {
        const MPIInfo mpi(_fv.Metrics().Communicator());
        const bool polynomial = bool(_fv.PolynomialReconstruction());
        std::optional<EquationTimeStepper::ActiveCheckpoint> active;
        MPI::LocalStage(mpi, [&]
                        {
            DNDS_check_throw_info(_timeStepper && (!polynomial || (_restoreReconstruction && _reconstructionBackup)) &&
                                      _trialEpoch.trial && state.stage >= 0 && state.iteration >= 0,
                                  "exact internal checkpoint has no active implicit trial");
            active.emplace(_timeStepper->CaptureActive(_trial)); });
        const bool ditr = _timeStepper->Method() == "ditr";
        MPI::LocalStage(mpi, [&]
                        { DNDS_check_throw_info(!ditr || (_coupledMiddle && _coupledMiddle->state),
                                                "DITR internal checkpoint lacks its midpoint reconstruction"); });
        const bool acceptedHistory = _timeStepper->Method() == "bdf2" && _timeStepper->HasHistory();
        const bool acceptedLimitedVR = polynomial && bool(_limitedVRBackup);
        const bool endpointLimitedVR = polynomial && bool(_fv.LimitedVRHistory(*_reconstruction));
        const bool middleLimitedVR = polynomial && ditr && bool(_fv.LimitedVRHistory(*_coupledMiddle->state));
        Json timePayloads = Json::array();
        for (index i = 0; i < index(active->fields.size()); ++i)
            timePayloads.push_back({{"name", active->names[size_t(i)]},
                                    {"storage", DescribeEquationStorage(active->fields[size_t(i)])}});
        Json timeState{{"schema_version", active->schemaVersion},
                       {"method", active->method},
                       {"pending_dt", active->pendingDt},
                       {"start_time", active->startTime},
                       {"pending_steady", active->pendingSteady},
                       {"stages_active", active->stagesActive},
                       {"stage_prepared", active->stagePrepared},
                       {"next_stage", active->nextStage},
                       {"equation", {{"inverse_dt", active->equation.inverseDt}, {"rhs_scale", active->equation.rhsScale}, {"time", active->equation.time}}},
                       {"bdf_history", active->bdfHistory},
                       {"bdf_previous_dt", active->bdfPreviousDt},
                       {"ditr_time", active->ditrTime},
                       {"ditr_dt", active->ditrDt},
                       {"payloads", std::move(timePayloads)}};
        const Json trialEpoch{{"accepted_step", _trialEpoch.acceptedStep},
                              {"trial", _trialEpoch.trial},
                              {"stage", _trialEpoch.stage},
                              {"coupled_stage", _trialEpoch.coupledStage},
                              {"coupling_iteration", _trialEpoch.couplingIteration},
                              {"geometry", _trialEpoch.geometry},
                              {"schema", _trialEpoch.schema}};
        Json metadata{{"schema_version", 1},
                      {"mesh_envelope", 2},
                      {"flow", EquationMetadata(_equation)},
                      {"reconstruction", _fv.ReconstructionMetadata()},
                      {"polynomial_reconstruction", polynomial},
                      {"time_method", _timeStepper->Method()},
                      {"history", acceptedHistory},
                      {"limited_vr_history", acceptedLimitedVR},
                      {"limited_vr_history_schema", 1},
                      {"frame", FrameMetadata(_fv, _time)},
                      {"restart_scope", "exact_internal"},
                      {"nonlinear_state", state.ToJson()},
                      {"trial_epoch", trialEpoch},
                      {"trial_dt", _trialDt},
                      {"progress_time", _progressTime},
                      {"time_state", timeState},
                      {"state_storage", DescribeEquationStorage(_accepted)},
                      {"endpoint_limited_vr_history", endpointLimitedVR},
                      {"middle_limited_vr_history", middleLimitedVR},
                      {"source_integration", _sourceStrang ? "strang" : "coupled"}};
        if (polynomial)
        {
            metadata["reconstruction_storage"] = DescribeEquationStorage(*_reconstructionBackup);
            metadata["endpoint_warm_storage"] = DescribeEquationStorage(_fv.ReconstructionGuess(*_reconstruction));
            metadata["endpoint_final_storage"] = DescribeEquationStorage(_fv.FluxReconstruction(*_reconstruction, *_workspace));
        }
        if (polynomial && ditr)
        {
            metadata["middle_warm_storage"] = DescribeEquationStorage(_fv.ReconstructionGuess(*_coupledMiddle->state));
            metadata["middle_final_storage"] = DescribeEquationStorage(_fv.FluxReconstruction(*_coupledMiddle->state, *_workspace));
        }
        auto serializer = std::make_shared<Serializer::SerializerH5>(mpi);
        serializer->OpenFile(path, false);
        serializer->WriteInt("cauchy_format", 5);
        serializer->WriteInt("cauchy_internal_format", 1);
        serializer->WriteString("resolved_config", resolved.dump());
        serializer->WriteString("metadata_json", metadata.dump());
        if (!outputController.is_null())
            serializer->WriteString("output_controller_json", outputController.dump());
        serializer->WriteString("domain_name", _name);
        serializer->WriteReal("accepted_time", _time);
        serializer->WriteReal("last_accepted_dt", _lastAcceptedDt);
        serializer->WriteIndex("accepted_step", index(_epoch.acceptedStep));
        Runtime::WriteArray(CellIds(*this), serializer, "cell_ids", mpi);
        std::optional<MeshEnvelope> envelope;
        MPI::LocalStage(mpi, [&]
                        { envelope.emplace(DescribeMesh(*this)); });
        Runtime::WriteArray(envelope->topology, serializer, "mesh_topology", mpi);
        Runtime::WriteArray(envelope->coordinates, serializer, "mesh_coordinates", mpi);
        Runtime::WriteArray(envelope->periodicTransforms, serializer, "periodic_transforms", mpi);
        WriteEquationArrays(_accepted, serializer, "state_field_", mpi);
        if (polynomial)
            WriteEquationArrays(*_reconstructionBackup, serializer, "reconstruction_field_", mpi);
        if (acceptedLimitedVR)
            Runtime::WriteArray(*_limitedVRBackup, serializer, "limited_vr_history", mpi);
        if (acceptedHistory)
        {
            serializer->WriteReal("history_dt", _timeStepper->PreviousDt());
            WriteEquationArrays(_timeStepper->PreviousState(), serializer, "history_field_", mpi);
        }
        for (index i = 0; i < index(active->fields.size()); ++i)
            WriteEquationArrays(active->fields[size_t(i)], serializer,
                                "internal_time_" + std::to_string(i) + "_field_", mpi);
        if (polynomial)
        {
            WriteEquationArrays(_fv.ReconstructionGuess(*_reconstruction), serializer,
                                "internal_endpoint_warm_field_", mpi);
            WriteEquationArrays(_fv.FluxReconstruction(*_reconstruction, *_workspace), serializer,
                                "internal_endpoint_final_field_", mpi);
            if (const auto history = _fv.LimitedVRHistory(*_reconstruction))
                Runtime::WriteArray(*history, serializer, "internal_endpoint_limited_vr_history", mpi);
            if (ditr)
            {
                WriteEquationArrays(_fv.ReconstructionGuess(*_coupledMiddle->state), serializer,
                                    "internal_middle_warm_field_", mpi);
                WriteEquationArrays(_fv.FluxReconstruction(*_coupledMiddle->state, *_workspace), serializer,
                                    "internal_middle_final_field_", mpi);
                if (const auto history = _fv.LimitedVRHistory(*_coupledMiddle->state))
                    Runtime::WriteArray(*history, serializer, "internal_middle_limited_vr_history", mpi);
            }
        }
        serializer->CloseFile();
    }
    void Domain::ReadCheckpoint(const std::string &path, const Json &resolved)
    {
        const MPIInfo mpi(_fv.Metrics().Communicator());
        bool internal = false;
        {
            auto serializer = std::make_shared<Serializer::SerializerH5>(mpi);
            serializer->OpenFile(path, true);
            internal = serializer->ListCurrentPath().count("cauchy_internal_format");
            serializer->CloseFile();
        }
        auto checkpoint = PrepareCheckpointRead(path, resolved);
        auto active = internal ? PrepareInternalCheckpointRead(path, resolved) : nullptr;
        ReserveCheckpointRead(*checkpoint);
        CommitCheckpointRead(*checkpoint);
        if (active)
            CommitInternalCheckpointRead(*active);
    }
    std::unique_ptr<Domain::CheckpointRead> Domain::PrepareCheckpointRead(const std::string &path, const Json &resolved)
    {
        const MPIInfo mpi(_fv.Metrics().Communicator());
        std::unique_ptr<CheckpointRead> checkpoint;
        MPI::LocalStage(mpi, [&]
                        { checkpoint = std::make_unique<CheckpointRead>(); });
        auto serializer = std::make_shared<Serializer::SerializerH5>(mpi);
        serializer->OpenFile(path, true);
        int version = 0;
        std::string config, name;
        real time = 0, lastDt = 0;
        index step = 0;
        serializer->ReadInt("cauchy_format", version);
        serializer->ReadString("resolved_config", config);
        serializer->ReadString("domain_name", name);
        serializer->ReadReal("accepted_time", time);
        serializer->ReadIndex("accepted_step", step);
        DNDS_check_throw_info(version >= 2 && version <= 5, "unsupported Cauchy checkpoint format");
        const auto rowCount = StoredRows(serializer, "cell_ids");
        const auto range = EvenSplitRange(mpi.rank, mpi.size, rowCount);
        const Serializer::ArrayGlobalOffset shard{range.second - range.first, range.first};
        auto ids = Runtime::ReadArray<index, Runtime::FixedRows<1>>(serializer, "cell_ids", mpi, shard);
        auto topology = Runtime::ReadArray<index>(serializer, "mesh_topology", mpi, shard);
        auto coordinates = Runtime::ReadArray<real>(serializer, "mesh_coordinates", mpi, shard);
        auto transforms = Runtime::ReadArray<real>(serializer, "periodic_transforms", mpi, {1, 0});
        std::optional<Runtime::Array<real>> flow;
        if (version < 5)
            flow.emplace(Runtime::ReadArray<real, Runtime::FixedRows<5>>(serializer, "flow", mpi, shard));
        Json metadata;
        if (version >= 4)
        {
            std::string serialized;
            serializer->ReadString("metadata_json", serialized);
            metadata = Json::parse(serialized, nullptr, false);
        }
        std::optional<Runtime::Array<real>> reconstruction;
        const auto entries = serializer->ListCurrentPath();
        if (entries.count("last_accepted_dt"))
            serializer->ReadReal("last_accepted_dt", lastDt);
        if (entries.count("reconstruction_guess"))
            reconstruction.emplace(Runtime::ReadArray<real>(serializer, "reconstruction_guess", mpi, shard));
        std::optional<Runtime::Array<real>> limitedVRHistory;
        if (entries.count("limited_vr_history"))
            limitedVRHistory.emplace(Runtime::ReadArray<real>(serializer, "limited_vr_history", mpi, shard));
        std::optional<Runtime::Array<real>> history;
        real historyDt = 0;
        if (version >= 4 && metadata.is_object() && metadata.value("history", false))
        {
            serializer->ReadReal("history_dt", historyDt);
            if (version < 5)
                history.emplace(Runtime::ReadArray<real, Runtime::FixedRows<5>>(serializer, "history_flow", mpi, shard));
        }
        std::optional<StateArchive> stateArchive, reconstructionArchive, historyArchive;
        if (version == 5)
        {
            stateArchive.emplace(StateArchive::Read(serializer, "state_field_", metadata.at("state_storage"), mpi, shard));
            if (metadata.contains("reconstruction_storage"))
                reconstructionArchive.emplace(StateArchive::Read(serializer, "reconstruction_field_", metadata.at("reconstruction_storage"), mpi, shard));
            if (metadata.value("history", false))
                historyArchive.emplace(StateArchive::Read(serializer, "history_field_", metadata.at("state_storage"), mpi, shard));
        }
        serializer->CloseFile();
        // Older BDF2 files carry the last physical dt with their history.
        // Other old files leave it unknown (zero): the first resumed step has
        // no growth cap, then its actual accepted dt establishes the reference.
        if (!entries.count("last_accepted_dt") && (history || historyArchive))
            lastDt = historyDt;
        DNDS_check_throw_info(std::isfinite(lastDt) && lastDt >= 0, "invalid stored last accepted timestep");
        if (std::isfinite(time) && time >= 0 && lastDt > time)
            DNDS_check_throw_info(lastDt - time <= 1024 * std::numeric_limits<real>::epsilon() * std::max(lastDt, time),
                                  "stored last accepted timestep exceeds the physical clock");
        if (entries.count("last_accepted_dt") && (history || historyArchive))
            DNDS_check_throw_info(lastDt == historyDt, "stored last accepted timestep conflicts with BDF history");
        auto storedConfig = Json::parse(config, nullptr, false);
        int valid = name == _name && !storedConfig.is_discarded() && CompatibleConfig(storedConfig) == CompatibleConfig(resolved) && std::isfinite(time) && time >= 0 && step >= 0 &&
                    (stateArchive ? stateArchive->Compatible(_trial) && stateArchive->fields[0].RowCount() == ids.RowCount()
                                  : flow && _trial.ComponentNumber() == 5 && _trial.PhysicalFieldNumber() == 1 &&
                                        flow->RowCount() == ids.RowCount() && flow->Layout().Width() == 5);
        valid = valid && (version < 4 || (metadata.is_object() && metadata.value("schema_version", 0) == 1 &&
                                          CompatibleEquationMetadata(metadata.value("flow", Json())) == CompatibleEquationMetadata(EquationMetadata(_equation))));
        valid = valid && CompatibleFrame(metadata, _fv, std::isfinite(time) ? time : 0);
        valid = valid && (!limitedVRHistory || ((reconstruction || reconstructionArchive) && limitedVRHistory->RowCount() == ids.RowCount() &&
                                                !limitedVRHistory->Layout().IsVariable() && limitedVRHistory->Layout().Width() == 2));
        valid = valid && (version < 4 || (metadata.is_object() && metadata.value("limited_vr_history", false) == bool(limitedVRHistory) &&
                                          (!limitedVRHistory || metadata.value("limited_vr_history_schema", 0) == 1)));
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, mpi.comm);
        if (!valid)
        {
            const auto differences = storedConfig.is_discarded() ? nlohmann::json() : nlohmann::json::diff(CompatibleConfig(storedConfig), CompatibleConfig(resolved));
            DNDS_check_throw_info(false, "Cauchy restart config/equation schema mismatch; normalized config differences: " + differences.dump());
        }
        const int meshEnvelope = version >= 4 ? metadata.value("mesh_envelope", 1) : 1;
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
        // Extended state shares the same mesh/ID validation and publication
        // transaction as historical flow5. No accepted field is changed until
        // means, optional reconstruction and BDF history have all been checked.
        if (stateArchive)
        {
            stateArchive->Restore(_trial, route, context);
            DNDS_check_throw_info(Admissible(_trial), "Cauchy restart contains inadmissible equation states");
            std::optional<EquationDofs> guess;
            MPI::LocalStage(mpi, [&]
                            { guess.emplace(EquationDofOperations::NewVector(_fv.ReconstructionGuess(*_reconstruction))); });
            const bool needsLimitedVRHistory = bool(_fv.LimitedVRHistory(*_reconstruction));
            int compatible = reconstructionArchive && reconstructionArchive->Compatible(*guess) &&
                             metadata.value("reconstruction", Json()) == _fv.ReconstructionMetadata() &&
                             (!needsLimitedVRHistory || limitedVRHistory);
            MPI_Allreduce(MPI_IN_PLACE, &compatible, 1, MPI_INT, MPI_MIN, mpi.comm);
            std::optional<Runtime::Array<real>> mappedLimitedVRHistory;
            if (compatible)
            {
                reconstructionArchive->Restore(*guess, route, context);
                if (needsLimitedVRHistory)
                {
                    MPI::LocalStage(mpi, [&]
                                    { mappedLimitedVRHistory.emplace(Runtime::RowLayout::Dense(_fv.Metrics().NumCell(), 2)); });
                    transfer(*limitedVRHistory, *mappedLimitedVRHistory);
                }
            }
            else
                EquationDofOperations::Fill(*guess, 0);
            std::unique_ptr<EquationTimeStepper> restoredStepper;
            if (_timeStepper)
            {
                MPI::LocalStage(mpi, [&]
                                { restoredStepper = _timeStepper->Fresh(); });
                if (!_sourceStrang && historyArchive && metadata.value("time_method", std::string{}) == restoredStepper->Method() && restoredStepper->Method() == "bdf2")
                {
                    std::optional<EquationDofs> previous;
                    MPI::LocalStage(mpi, [&]
                                    {
                        DNDS_check_throw_info(std::isfinite(historyDt) && historyDt > 0, "invalid equation restart history timestep");
                        previous.emplace(EquationDofOperations::NewVector(_trial)); });
                    historyArchive->Restore(*previous, route, context);
                    DNDS_check_throw_info(Admissible(*previous), "inadmissible equation restart history");
                    MPI::LocalStage(mpi, [&]
                                    { restoredStepper->RestoreHistory(*previous, historyDt); });
                }
            }
            checkpoint->time = time;
            checkpoint->lastDt = lastDt;
            checkpoint->epoch.acceptedStep = uint64_t(step);
            checkpoint->stepper = std::move(restoredStepper);
            checkpoint->equationGuess = std::move(guess);
            checkpoint->limitedVRHistory = std::move(mappedLimitedVRHistory);
            return checkpoint;
        }
        // Existing trial allocations and binding identities remain stable.
        auto &trial = _trial.CoupledField();
        transfer(*flow, trial.Owned());
        DNDS_check_throw_info(Admissible(_trial), "Cauchy restart contains inadmissible states");
        std::optional<EquationDofs> guess;
        MPI::LocalStage(mpi, [&]
                        { guess.emplace(EquationDofOperations::NewVector(_fv.ReconstructionGuess(*_reconstruction))); });
        auto &guessArray = guess->CoupledField().Owned();
        int compatibleReconstruction = bool(reconstruction) && (version < 4 || metadata.value("reconstruction", Json()) == _fv.ReconstructionMetadata());
        const bool needsLimitedVRHistory = _fv.LimitedVRHistory(*_reconstruction).has_value();
        compatibleReconstruction = compatibleReconstruction && (!needsLimitedVRHistory || limitedVRHistory);
        if (reconstruction)
            for (index iRow = 0; iRow < reconstruction->RowCount(); ++iRow)
                compatibleReconstruction = compatibleReconstruction && reconstruction->RowSize(iRow) == guessArray.Layout().Width();
        MPI_Allreduce(MPI_IN_PLACE, &compatibleReconstruction, 1, MPI_INT, MPI_MIN, mpi.comm);
        std::optional<Runtime::Array<real>> mappedLimitedVRHistory;
        if (compatibleReconstruction)
        {
            transfer(*reconstruction, guessArray);
            if (needsLimitedVRHistory)
            {
                MPI::LocalStage(mpi, [&]
                                { mappedLimitedVRHistory.emplace(Runtime::RowLayout::Dense(_fv.Metrics().NumCell(), 2)); });
                transfer(*limitedVRHistory, *mappedLimitedVRHistory);
            }
        }
        else
        {
            auto write = guessArray.Write();
            const auto output = write.View<Runtime::UniformTableRows>();
            for (index iCell = 0; iCell < output.RowCount(); ++iCell)
                for (index j = 0; j < output.RowSize(iCell); ++j)
                    output(iCell, j) = 0;
        }
        std::unique_ptr<EquationTimeStepper> restoredStepper;
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
                    restoredHistory.emplace(_fv.Metrics().BuildField(_accepted.CoupledField().Owned().Layout())); });
                transfer(*history, restoredHistory->Owned());
                const auto restoredState = _trial.RebindCoupled(*restoredHistory);
                DNDS_check_throw_info(Admissible(restoredState), "inadmissible Cauchy restart history");
                MPI::LocalStage(mpi, [&]
                                { restoredStepper->RestoreHistory(restoredState, historyDt); });
            }
        }
        checkpoint->time = time;
        checkpoint->lastDt = lastDt;
        checkpoint->epoch.acceptedStep = uint64_t(step);
        checkpoint->stepper = std::move(restoredStepper);
        checkpoint->equationGuess = std::move(guess);
        checkpoint->limitedVRHistory = std::move(mappedLimitedVRHistory);
        return checkpoint;
    }
    std::unique_ptr<Domain::InternalCheckpointRead> Domain::PrepareInternalCheckpointRead(const std::string &path, const Json &resolved)
    {
        const MPIInfo mpi(_fv.Metrics().Communicator());
        auto result = std::make_unique<InternalCheckpointRead>();
        auto serializer = std::make_shared<Serializer::SerializerH5>(mpi);
        serializer->OpenFile(path, true);
        int format = 0;
        serializer->ReadInt("cauchy_internal_format", format);
        std::string serializedMetadata, serializedConfig;
        serializer->ReadString("metadata_json", serializedMetadata);
        serializer->ReadString("resolved_config", serializedConfig);
        const auto metadata = Json::parse(serializedMetadata, nullptr, false);
        const auto storedConfig = Json::parse(serializedConfig, nullptr, false);
        const bool polynomial = bool(_fv.PolynomialReconstruction());
        int valid = format == 1 && metadata.is_object() && storedConfig.is_object() &&
                    metadata.value("restart_scope", std::string{}) == "exact_internal" &&
                    CompatibleConfig(storedConfig) == CompatibleConfig(resolved) && _timeStepper &&
                    metadata.value("time_method", std::string{}) == _timeStepper->Method() &&
                    metadata.value("polynomial_reconstruction", true) == polynomial &&
                    metadata.value("reconstruction", Json()) == _fv.ReconstructionMetadata();
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, mpi.comm);
        DNDS_check_throw_info(valid, "Cauchy exact internal checkpoint metadata differs");
        const auto rowCount = StoredRows(serializer, "cell_ids");
        const auto range = EvenSplitRange(mpi.rank, mpi.size, rowCount);
        const Serializer::ArrayGlobalOffset shard{range.second - range.first, range.first};
        auto ids = Runtime::ReadArray<index, Runtime::FixedRows<1>>(serializer, "cell_ids", mpi, shard);
        auto context = std::make_shared<Runtime::CommunicationContext>(mpi.comm);
        const auto route = Runtime::BuildRowTransferById(context, ids, CellIds(*this), true);
        const auto transfer = [&](const auto &source, auto &destination)
        {
            using T = typename std::decay_t<decltype(source)>::value_type;
            Runtime::PayloadPlan(route, source.Layout(), destination.Layout(), Runtime::ElementSchema::For<T>())
                .Bind(source, destination, Runtime::TransferTransport::Direct,
                      {Runtime::TransferMode::Transient, context, 0})
                .Begin()
                .Wait();
        };
        const auto &time = metadata.at("time_state");
        result->time.schemaVersion = time.at("schema_version").get<int>();
        result->time.method = time.at("method").get<std::string>();
        result->time.pendingDt = time.at("pending_dt").get<real>();
        result->time.startTime = time.at("start_time").get<real>();
        result->time.pendingSteady = time.at("pending_steady").get<bool>();
        result->time.stagesActive = time.at("stages_active").get<bool>();
        result->time.stagePrepared = time.at("stage_prepared").get<bool>();
        result->time.nextStage = time.at("next_stage").get<int>();
        result->time.equation = {time.at("equation").at("inverse_dt").get<real>(),
                                 time.at("equation").at("rhs_scale").get<real>(),
                                 time.at("equation").at("time").get<real>()};
        result->time.bdfHistory = time.at("bdf_history").get<bool>();
        result->time.bdfPreviousDt = time.at("bdf_previous_dt").get<real>();
        result->time.ditrTime = time.at("ditr_time").get<real>();
        result->time.ditrDt = time.at("ditr_dt").get<real>();
        std::set<std::string> payloadNames;
        index payloadOrdinal = 0;
        for (const auto &payload : time.at("payloads"))
        {
            const auto name = payload.at("name").get<std::string>();
            DNDS_check_throw_info(!name.empty() && payloadNames.insert(name).second,
                                  "duplicate exact time-checkpoint payload");
            auto archive = StateArchive::Read(serializer,
                                              "internal_time_" + std::to_string(payloadOrdinal) + "_field_",
                                              payload.at("storage"), mpi, shard);
            std::optional<EquationDofs> field;
            MPI::LocalStage(mpi, [&]
                            {
                field.emplace(EquationDofOperations::NewVector(_trial));
                DNDS_check_throw_info(archive.Compatible(*field), "time-checkpoint equation payload differs"); });
            archive.Restore(*field, route, context);
            result->time.names.push_back(name);
            result->time.fields.push_back(std::move(*field));
            ++payloadOrdinal;
        }
        const auto restoreReconstruction = [&](const char *prefix, const Json &description)
        {
            auto archive = StateArchive::Read(serializer, prefix, description, mpi, shard);
            std::optional<EquationDofs> field;
            MPI::LocalStage(mpi, [&]
                            {
                field.emplace(EquationDofOperations::NewVector(_fv.ReconstructionGuess(*_reconstruction)));
                DNDS_check_throw_info(archive.Compatible(*field), "internal reconstruction payload differs"); });
            archive.Restore(*field, route, context);
            return std::move(*field);
        };
        if (polynomial)
        {
            result->endpointWarm.emplace(restoreReconstruction("internal_endpoint_warm_field_", metadata.at("endpoint_warm_storage")));
            result->endpointFinal.emplace(restoreReconstruction("internal_endpoint_final_field_", metadata.at("endpoint_final_storage")));
        }
        const auto entries = serializer->ListCurrentPath();
        const auto restoreHistory = [&](const std::string &name) -> std::optional<Runtime::Array<real>>
        {
            if (!entries.count(name))
                return std::nullopt;
            auto source = Runtime::ReadArray<real>(serializer, name, mpi, shard);
            std::optional<Runtime::Array<real>> destination;
            MPI::LocalStage(mpi, [&]
                            { destination.emplace(Runtime::RowLayout::Dense(_fv.Metrics().NumCell(), 2)); });
            transfer(source, *destination);
            return destination;
        };
        if (polynomial)
            result->endpointHistory = restoreHistory("internal_endpoint_limited_vr_history");
        result->ditr = result->time.method == "ditr";
        if (polynomial && result->ditr)
        {
            result->middleWarm.emplace(restoreReconstruction("internal_middle_warm_field_", metadata.at("middle_warm_storage")));
            result->middleFinal.emplace(restoreReconstruction("internal_middle_final_field_", metadata.at("middle_final_storage")));
            result->middleHistory = restoreHistory("internal_middle_limited_vr_history");
        }
        result->nonlinear = NonlinearSolveState::FromJson(metadata.at("nonlinear_state"));
        const auto &epoch = metadata.at("trial_epoch");
        result->trialEpoch.acceptedStep = epoch.at("accepted_step").get<uint64_t>();
        result->trialEpoch.trial = epoch.at("trial").get<uint64_t>();
        result->trialEpoch.stage = epoch.at("stage").get<uint64_t>();
        result->trialEpoch.coupledStage = epoch.at("coupled_stage").get<uint64_t>();
        result->trialEpoch.couplingIteration = epoch.at("coupling_iteration").get<uint64_t>();
        result->trialEpoch.geometry = epoch.at("geometry").get<uint64_t>();
        result->trialEpoch.schema = epoch.at("schema").get<uint64_t>();
        result->trialDt = metadata.at("trial_dt").get<real>();
        result->progressTime = metadata.at("progress_time").get<real>();
        serializer->CloseFile();
        MPI::LocalStage(mpi, [&]
                        {
            DNDS_check_throw_info(result->trialEpoch.trial > 0 && std::isfinite(result->trialDt) &&
                                      result->trialDt >= 0 && std::isfinite(result->progressTime),
                                  "invalid exact internal checkpoint cursor");
            const auto found = std::find(result->time.names.begin(), result->time.names.end(), "trial");
            DNDS_check_throw_info(found != result->time.names.end() &&
                                      Admissible(result->time.fields[size_t(found - result->time.names.begin())]),
                                  "exact internal checkpoint trial is inadmissible");
            auto validationTrial = EquationDofOperations::NewVector(_trial);
            auto validationStepper = _timeStepper->Fresh();
            validationStepper->RestoreActive(result->time, validationTrial); });
        return result;
    }
    void Domain::ReserveCheckpointRead(CheckpointRead &checkpoint, bool collectiveChecks)
    {
        const auto reserve = [&]
        {
            DNDS_check_throw_info(!checkpoint.committed && !checkpoint.destination, "checkpoint publication was already reserved or committed");
            checkpoint.source.emplace(_trial.ReadOwned());
            checkpoint.destination.emplace(_accepted.WriteOwned());
            for (index iBlock = 0; iBlock < _accepted.BlockNumber(); ++iBlock)
            {
                checkpoint.input.push_back(checkpoint.source->Block(iBlock).owned);
                checkpoint.output.push_back(checkpoint.destination->Block(iBlock));
            } };
        if (collectiveChecks)
            MPI::LocalStage(MPIInfo(_fv.Metrics().Communicator()), reserve);
        else
            reserve();
        if (_fv.PolynomialReconstruction())
        {
            DNDS_check_throw_info(bool(checkpoint.equationGuess),
                                  "checkpoint lacks a unified equation reconstruction candidate");
            checkpoint.reconstruction.emplace(_fv.ReserveReconstructionRestore(
                _reconstruction, *checkpoint.equationGuess,
                checkpoint.limitedVRHistory ? std::optional(std::cref(*checkpoint.limitedVRHistory)) : std::nullopt,
                collectiveChecks));
        }
    }
    void Domain::CommitCheckpointRead(CheckpointRead &checkpoint)
    {
        // Every participating domain/rank has reserved before this point.
        // The retained views and finite scalar copies need no allocation, MPI
        // or further user-input validation during publication.
        if (checkpoint.reconstruction)
            checkpoint.reconstruction->Commit();
        checkpoint.committed = true;
        for (index iBlock = 0; iBlock < index(checkpoint.input.size()); ++iBlock)
        {
            const auto input = checkpoint.input[iBlock];
            const auto output = checkpoint.output[iBlock];
            for (index iCell = 0; iCell < output.RowCount(); ++iCell)
                for (index j = 0; j < output.ComponentNumber(); ++j)
                    output(iCell, j) = input(iCell, j);
        }
        checkpoint.input.clear();
        checkpoint.output.clear();
        checkpoint.source.reset();
        checkpoint.destination.reset();
        checkpoint.reconstruction.reset();
        _time = checkpoint.time;
        _epoch = checkpoint.epoch;
        _trialReady = _restoreReconstruction = _coordinatedPrepared = false;
        _stageEquation.reset();
        if (checkpoint.stepper)
            _timeStepper.swap(checkpoint.stepper);
        _lastAcceptedDt = checkpoint.lastDt;
        _nextTrial = 0;
    }
    void Domain::CommitInternalCheckpointRead(InternalCheckpointRead &checkpoint)
    {
        const bool polynomial = bool(_fv.PolynomialReconstruction());
        DNDS_check_throw_info(_timeStepper &&
                                  (!polynomial || (checkpoint.endpointWarm && checkpoint.endpointFinal)) &&
                                  checkpoint.trialEpoch.acceptedStep == _epoch.acceptedStep,
                              "internal checkpoint cannot follow the restored accepted state");
        if (polynomial)
        {
            const auto &acceptedGuess = _fv.ReconstructionGuess(*_reconstruction);
            if (!_reconstructionBackup)
                _reconstructionBackup.emplace(EquationDofOperations::NewVector(acceptedGuess));
            EquationDofOperations::Copy(*_reconstructionBackup, acceptedGuess);
            if (const auto history = _fv.LimitedVRHistory(*_reconstruction))
            {
                if (!_limitedVRBackup)
                    _limitedVRBackup.emplace(history->Layout());
                _limitedVRBackup->CopyFrom(*history).Wait();
            }
            else
                _limitedVRBackup.reset();
        }
        else
        {
            _reconstructionBackup.reset();
            _limitedVRBackup.reset();
        }
        _timeStepper->RestoreActive(checkpoint.time, _trial);
        if (polynomial)
            _fv.RestoreReconstructionState(*_reconstruction, *checkpoint.endpointWarm,
                                           *checkpoint.endpointFinal,
                                           checkpoint.endpointHistory ? std::optional(std::cref(*checkpoint.endpointHistory)) : std::nullopt);
        if (checkpoint.ditr)
        {
            DNDS_check_throw_info(!polynomial || (checkpoint.middleWarm && checkpoint.middleFinal),
                                  "DITR internal checkpoint lacks midpoint reconstruction");
            auto &middle = _timeStepper->CoupledMethod().IntermediateState();
            _coupledMiddle = _coupledFactory(middle);
            _coupledMiddleOwner = middle.PhysicalField(0).Owned().SharedStorage();
            if (polynomial)
                _fv.RestoreReconstructionState(*_coupledMiddle->state, *checkpoint.middleWarm,
                                               *checkpoint.middleFinal,
                                               checkpoint.middleHistory ? std::optional(std::cref(*checkpoint.middleHistory)) : std::nullopt);
            _coupledNonlinearState = checkpoint.nonlinear;
        }
        else
            _implicit->RestoreState(checkpoint.nonlinear);
        _trialEpoch = checkpoint.trialEpoch;
        _trialDt = checkpoint.trialDt;
        _progressTime = checkpoint.progressTime;
        _trialReady = false;
        _restoreReconstruction = polynomial;
        _internalResume = true;
        _coordinatedPrepared = false;
        _stageEquation.reset();
        _nextTrial = std::max(_nextTrial, _trialEpoch.trial);
        _lastImplicit.clear();
        _lastCoupledIterations = 0;
        _lastCoupledTargetRatio = 0;
    }
}
