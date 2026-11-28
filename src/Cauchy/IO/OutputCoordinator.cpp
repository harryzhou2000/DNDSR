#include "Cauchy/IO/OutputCoordinator.hpp"

#include "Cauchy/IO/AsyncOutput.hpp"
#include "Cauchy/IO/BoundaryName.hpp"
#include "Cauchy/IO/ProgressOutput.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <filesystem>
#include <fstream>

namespace DNDS::Cauchy
{
    namespace
    {
        Json BoundaryArtifact(const std::string &prefix, bool facialFluxes, real observationTime)
        {
            Json artifact{{"kind", "boundary"}, {"path", prefix + "_bnd.vtkhdf"}, {"schema", "dndsr.cauchy.boundary.v1"}, {"facialFluxes", {{"available", facialFluxes}}}};
            if (facialFluxes)
                artifact["facialFluxes"].update({{"observationTime", observationTime},
                                                 {"sign", "negative_outward_rhs_flux"},
                                                 {"normalization", "rhs_quadrature_area_average"}});
            return artifact;
        }

        Json CsvRowToJson(const CsvRow &row)
        {
            Json values = Json::object();
            const auto &columns = row.Columns();
            const auto &cells = row.Cells();
            for (size_t i = 0; i < cells.size(); ++i)
                std::visit([&](const auto &value)
                           {
                    using T = std::decay_t<decltype(value)>;
                    if constexpr (!std::is_same_v<T, std::monostate>)
                        values[columns[i].name] = value; }, cells[i]);
            return values;
        }

        Json GatherJsonArrays(const MPIInfo &mpi, const Json &local)
        {
            const auto encoded = local.dump();
            DNDS_check_throw_info(encoded.size() <= size_t(INT_MAX),
                                  "Cauchy manifest contribution is too large");
            const int count = int(encoded.size());
            std::vector<int> counts(mpi.rank == 0 ? size_t(mpi.size) : 0);
            MPI_Gather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0,
                       mpi.comm);
            std::vector<int> offsets;
            std::vector<char> bytes;
            if (mpi.rank == 0)
            {
                offsets.resize(size_t(mpi.size));
                int total = 0;
                for (int rank = 0; rank < mpi.size; ++rank)
                {
                    offsets[rank] = total;
                    DNDS_check_throw_info(
                        counts[rank] <= INT_MAX - total,
                        "Cauchy manifest gather is too large");
                    total += counts[rank];
                }
                bytes.resize(size_t(total));
            }
            MPI_Gatherv(encoded.data(), count, MPI_CHAR, bytes.data(),
                        counts.data(), offsets.data(), MPI_CHAR, 0, mpi.comm);
            Json result = Json::array();
            if (mpi.rank == 0)
                for (int rank = 0; rank < mpi.size; ++rank)
                {
                    const auto contribution = Json::parse(
                        bytes.data() + offsets[rank],
                        bytes.data() + offsets[rank] + counts[rank]);
                    for (const auto &entry : contribution)
                        result.push_back(entry);
                }
            return result;
        }

        Json BroadcastJson(const MPIInfo &mpi, const Json &rootPayload)
        {
            std::string encoded;
            int length = 0;
            if (mpi.rank == 0)
            {
                encoded = rootPayload.dump();
                DNDS_check_throw_info(encoded.size() <= size_t(INT_MAX),
                                      "Cauchy control broadcast payload is too large");
                length = int(encoded.size());
            }
            MPI_Bcast(&length, 1, MPI_INT, 0, mpi.comm);
            if (mpi.rank != 0)
                encoded.resize(size_t(length));
            if (length > 0)
                MPI_Bcast(encoded.data(), length, MPI_CHAR, 0, mpi.comm);
            return length > 0 ? Json::parse(encoded) : Json();
        }
    }

    struct OutputCoordinator::Data
    {
        Simulation &simulation;
        const SimulationConfig &config;
        MPIInfo mpi;
        real started;
        RunControl *control;
        std::function<void()> pollInput;
        const std::chrono::steady_clock::time_point progressStart = std::chrono::steady_clock::now();
        const real progressOffset = MPI_Wtime() - started;
        std::optional<real> stepStarted;
        real stepWallSeconds = 0;
        OutputSettings output;
        TimeSettings time;
        std::vector<ImplicitSettings> domainImplicit;
        Json currentConfigJson;
        index domains;
        OutputPlan plan;
        OutputControllerState state;
        std::shared_ptr<OperationalLogger> logger;
        std::vector<std::unique_ptr<ProgressOutputController>> progress;
        std::vector<std::vector<BoundaryIntegrationSpec>> boundarySpecs;
        std::vector<std::vector<int>> boundaryZones;
        std::vector<real> baselines;
        uint64_t nextCheckpointTime = 0, nextVisualizationTime = 0;
        index internalCheckpointInterval = 0,
              internalVisualizationInterval = 0;
        bool continueOutput = false, finalized = false, allocationReportWritten = false;
        bool iterationStop = false;
        bool warnedMultiDomainIterationStop = false;
        std::optional<LiveSnapshotRequest> pendingInternalSnapshot;
        index lastObservedStage = -1, lastObservedIteration = -1;
        std::string stopCheckpoint;
        std::unique_ptr<AsyncOutput> asynchronous;

        Data(Simulation &simulationIn, const SimulationConfig &configIn,
             MPIInfo mpiIn, real startedIn, RunControl *controlIn,
             std::function<void()> pollInputIn)
            : simulation(simulationIn), config(configIn), mpi(mpiIn),
              started(startedIn), control(controlIn), pollInput(std::move(pollInputIn)),
              output(config.Output()), time(config.Time()),
              currentConfigJson(config.Root()),
              domains(index(config.Domains().size())), plan(output, domains),
              progress(size_t(domains)), boundarySpecs(size_t(domains)),
              boundaryZones(size_t(domains)), baselines(size_t(domains), 0),
              internalCheckpointInterval(output.implicit.checkpointInterval),
              internalVisualizationInterval(
                  output.implicit.visualizationInterval < 0
                      ? output.implicit.checkpointInterval
                      : output.implicit.visualizationInterval)
        {
            domainImplicit.reserve(size_t(domains));
            for (index iDomain = 0; iDomain < domains; ++iDomain)
            {
                domainImplicit.push_back(config.Domains()[size_t(iDomain)].settings.implicit);
                boundarySpecs[size_t(iDomain)] =
                    ResolveBoundaryIntegrationSpecs(config.Domains()[size_t(iDomain)].settings);
                for (const auto &spec : boundarySpecs[size_t(iDomain)])
                    boundaryZones[size_t(iDomain)].push_back(spec.zone);
            }
            // A dislocated domain root need not be world rank zero. Let each
            // actual history writer print its own domain's progress without
            // gathering residuals or adding collectives to the output path.
            bool domainProgressWriter = false;
            for (index iDomain = 0; iDomain < domains; ++iDomain)
                domainProgressWriter |= simulation.Local(iDomain) &&
                                        simulation.Placement().DomainRank(iDomain) == 0;
            logger = std::make_shared<OperationalLogger>(output, mpi.rank, domainProgressWriter);
            const auto &restored = simulation.RestoredOutputController();
            continueOutput = restored && restored->is_object() &&
                             restored->value("prefix", std::string{}) ==
                                 output.prefix;
            state = restored ? plan.RestoreState(*restored, simulation.Time())
                             : plan.InitialState(simulation.Time());
            if (state.boundaryHistory.size() < size_t(domains))
                state.boundaryHistory.resize(size_t(domains));
            nextCheckpointTime = state.nextCheckpointTime;
            nextVisualizationTime = state.nextVisualizationTime;
            InitializeFiles();
            if (output.asynchronous)
                asynchronous = std::make_unique<AsyncOutput>(simulation);
            if (!continueOutput)
                Record(0, 0, {}, false, false, true);
            if (output.diagnostics.allocationWarmupSteps >= 0)
                MaybeReportAllocations();
            // Publish callbacks only after fallible output/resource setup.
            // A failed constructor has no Data destructor to withdraw them.
            InstallObservers();
            if (control)
                simulation._stopAfterCorrection = [this]
                {
                    iterationStop = PollControl(ControlBoundary::Iteration, false) == RunStop::Requested;
                    return iterationStop;
                };
            PublishStatus("running");
        }

        ~Data() noexcept
        {
            simulation._stopAfterCorrection = {};
            if (asynchronous)
            {
                if (std::uncaught_exceptions())
                    MPI::AbortIfSolverFailure("solver unwinding with asynchronous output");
                try
                {
                    asynchronous->Drain();
                }
                catch (...)
                {
                    MPI::AbortIfSolverFailure("asynchronous output drain failed during teardown");
                }
            }
            // Domain callbacks capture this owner. Withdraw them before logger,
            // CSV and scheduling state can disappear.
            try
            {
                Visit([&](index, Domain &domain)
                      {
                          domain.SetProgressObserver({});
                          domain.SetQuiescentObserver({}); });
            }
            catch (...)
            {
                MPI::AbortIfSolverFailure("output observer cleanup failed during teardown");
            }
        }

        void PublishStatus(const std::string &runState, const Json &extra = Json::object()) const
        {
            if (!control || mpi.rank != 0)
                return;
            Json status{
                {"state", runState},
                {"step", simulation.Step()},
                {"stage", lastObservedStage},
                {"iteration", lastObservedIteration},
                {"time", simulation.Time()},
                {"wallSeconds", MPI_Wtime() - started},
                {"steady", time.steady},
                {"targetSteps", time.steps},
                {"physicalTimeTarget", time.tEnd},
                {"wallLimitSeconds", time.wallSeconds},
                {"domains", domains},
                {"prefix", output.prefix},
                {"runManifest", plan.RunManifestPath()}};
            if (extra.is_object())
                status.update(extra);
            control->UpdateStatus(status);
        }

        void ApplyLiveConfigRequests(const std::vector<LiveConfigRequest> &requests,
                                     ControlBoundary boundary)
        {
            const uint64_t curStep = boundary == ControlBoundary::Iteration
                                         ? simulation.Step() + 1
                                         : simulation.Step();
            const index curStage = boundary == ControlBoundary::Iteration ? lastObservedStage : -1;
            const index curIter = boundary == ControlBoundary::Iteration ? lastObservedIteration : -1;

            for (const auto &req : requests)
            {
                LiveConfigCategory category = LiveConfigCategory::Output;
                Json normalized = Json::object();
                const auto [ok, error] = RunControl::ValidatePatch(currentConfigJson, req.patch, category, normalized);
                if (!ok)
                {
                    if (control && mpi.rank == 0)
                    {
                        control->RecordRejectedConfig(req.revision, error, curStep, curStage, curIter);
                        logger->Warn(fmt::format("Rejected live config revision {}: {}", req.revision, error));
                    }
                    continue;
                }

                const real prevCkptTime = output.checkpointTimeInterval;
                const real prevVisTime = plan.VisualizationTimeInterval();
                bool outputChanged = false;
                std::vector<bool> domainImplicitChanged(size_t(domains), false);
                std::vector<bool> domainPseudoCFLChanged(size_t(domains), false);

                for (const auto &[pointer, value] : normalized.items())
                {
                    (void)value;
                    if (pointer.rfind("/output/", 0) == 0)
                        outputChanged = true;
                    else if (pointer.rfind("/domains/", 0) == 0)
                    {
                        const auto rest = pointer.substr(std::string("/domains/").size());
                        const auto slash = rest.find('/');
                        if (slash != std::string::npos)
                        {
                            const auto d = index(std::stoll(rest.substr(0, slash)));
                            if (d >= 0 && d < domains)
                            {
                                domainImplicitChanged[size_t(d)] = true;
                                if (pointer == "/domains/" + std::to_string(d) + "/settings/implicit/pseudoCFL")
                                    domainPseudoCFLChanged[size_t(d)] = true;
                            }
                        }
                    }
                }

                currentConfigJson = RunControl::ApplyNormalizedPatchToJson(currentConfigJson, normalized);
                time = currentConfigJson.at("time").get<TimeSettings>();
                output = currentConfigJson.at("output").get<OutputSettings>();

                if (outputChanged)
                {
                    plan = OutputPlan(output, domains);
                    internalCheckpointInterval = output.implicit.checkpointInterval;
                    internalVisualizationInterval =
                        output.implicit.visualizationInterval < 0
                            ? output.implicit.checkpointInterval
                            : output.implicit.visualizationInterval;
                    if (output.checkpointTimeInterval != prevCkptTime)
                        nextCheckpointTime = OutputPlan::NextTimeOrdinal(simulation.Time(), output.checkpointTimeInterval);
                    if (plan.VisualizationTimeInterval() != prevVisTime)
                        nextVisualizationTime = OutputPlan::NextTimeOrdinal(simulation.Time(), plan.VisualizationTimeInterval());
                    Visit([&](index iDomain, Domain &domain)
                          {
                              progress[size_t(iDomain)]->UpdateSettings(output);
                              domain.UpdateProgressCadence(
                                  output.historyStateBounds ? output.implicit.historyInterval : 0,
                                  boundaryZones[size_t(iDomain)].empty() ? 0 : output.implicit.historyInterval,
                                  output.implicit.consoleInterval > 0); });
                }

                for (index d = 0; d < domains; ++d)
                {
                    if (!domainImplicitChanged[size_t(d)])
                        continue;
                    domainImplicit[size_t(d)] =
                        currentConfigJson["domains"][size_t(d)]["settings"]["implicit"].get<ImplicitSettings>();
                    simulation.ApplyLiveImplicitSettings(d, domainImplicit[size_t(d)], domainPseudoCFLChanged[size_t(d)]);
                }

                if (control && mpi.rank == 0)
                {
                    control->RecordAppliedConfig(req.revision, currentConfigJson, curStep, curStage, curIter);
                    logger->Info(fmt::format("Applied live {} config revision {} at {} boundary: {}",
                                             ToString(category), req.revision, ToString(boundary), normalized.dump()));
                }
            }
            PublishStatus("running");
        }

        RunStop PollControl(ControlBoundary boundary, bool wallLimitReached)
        {
            if (pollInput)
                pollInput();
            const bool allowIterationStop =
                domains == 1 &&
                simulation.Placement().Group(0).ranks.size() == size_t(mpi.size);
            bool localRequestedStop = false;
            if (control)
            {
                if (boundary == ControlBoundary::Step)
                    localRequestedStop = control->StopRequested(ControlBoundary::Step, time.steady);
                else if (allowIterationStop)
                    localRequestedStop = control->StopRequested(ControlBoundary::Iteration, time.steady);
                else if (control->StopRequested(ControlBoundary::Iteration, time.steady) &&
                         mpi.rank == 0 && !warnedMultiDomainIterationStop)
                {
                    warnedMultiDomainIterationStop = true;
                    logger->Warn("Iteration-boundary stop requires a single full-rank domain; deferring stop to step boundary");
                }
            }
            const int localStop = localRequestedStop ? int(RunStop::Requested)
                                  : wallLimitReached ? int(RunStop::WallLimit)
                                                     : int(RunStop::None);

            std::vector<LiveConfigRequest> localPatches;
            std::optional<LiveSnapshotRequest> localSnapshot;
            if (control && mpi.rank == 0)
            {
                localPatches = control->TakePendingConfigPatches(boundary);
                localSnapshot = control->TakePendingSnapshot(boundary);
            }

            int flags[3]{localStop, int(!localPatches.empty()), int(localSnapshot.has_value())};
            simulation._controlReductions.Reduce<Runtime::Reduction::Max>(SpanView<int>{flags, 3});

            if (flags[1] != 0)
            {
                Json encoded = Json::array();
                if (mpi.rank == 0)
                    for (const auto &req : localPatches)
                        encoded.push_back({{"revision", req.revision},
                                           {"applyAt", ToString(req.applyAt)},
                                           {"category", ToString(req.category)},
                                           {"patch", req.patch}});
                const Json broadcast = BroadcastJson(mpi, encoded);
                std::vector<LiveConfigRequest> agreed;
                agreed.reserve(broadcast.size());
                for (const auto &item : broadcast)
                {
                    LiveConfigRequest req;
                    req.revision = item.at("revision").get<uint64_t>();
                    req.applyAt = ParseControlBoundary(item.at("applyAt").get<std::string>()).value_or(boundary);
                    req.category = item.at("category").get<std::string>() == "steering"
                                       ? LiveConfigCategory::Steering
                                       : LiveConfigCategory::Output;
                    req.patch = item.at("patch");
                    agreed.push_back(std::move(req));
                }
                ApplyLiveConfigRequests(agreed, boundary);
            }

            if (flags[2] != 0)
            {
                int64_t snapEncoded[3]{};
                if (mpi.rank == 0 && localSnapshot)
                {
                    snapEncoded[0] = int64_t(localSnapshot->id);
                    snapEncoded[1] = int64_t(localSnapshot->checkpoint);
                    snapEncoded[2] = int64_t(localSnapshot->visualization);
                }
                MPI_Bcast(snapEncoded, 3, MPI_INT64_T, 0, mpi.comm);
                LiveSnapshotRequest agreedSnap{uint64_t(snapEncoded[0]), boundary,
                                               snapEncoded[1] != 0, snapEncoded[2] != 0};
                if (boundary == ControlBoundary::Step)
                {
                    const auto token = OutputPlan::StepToken(simulation.Step()) +
                                       "-live" + std::to_string(agreedSnap.id);
                    WriteSnapshot(token, agreedSnap.checkpoint, agreedSnap.visualization);
                    if (control && mpi.rank == 0)
                        control->RecordAppliedSnapshot(agreedSnap.id, simulation.Step(), -1, -1);
                }
                else
                {
                    if (agreedSnap.checkpoint && domains > 1)
                    {
                        if (mpi.rank == 0)
                            logger->Warn("Internal iteration checkpoint requires a single domain; emitting visualization only");
                        agreedSnap.checkpoint = false;
                    }
                    if (agreedSnap.checkpoint || agreedSnap.visualization)
                        pendingInternalSnapshot = agreedSnap;
                }
            }

            const auto stop = RunStop(flags[0]);
            if (stop == RunStop::Requested && control && mpi.rank == 0)
                control->RecordStopApplied(
                    boundary == ControlBoundary::Iteration ? simulation.Step() + 1 : simulation.Step(),
                    boundary == ControlBoundary::Iteration ? lastObservedStage : -1,
                    boundary == ControlBoundary::Iteration ? lastObservedIteration : -1,
                    ToString(boundary));
            return stop;
        }

        RunStop PollStop(bool wallLimitReached)
        {
            return PollControl(ControlBoundary::Step, wallLimitReached);
        }

        template <class Action>
        void Visit(Action &&action)
        {
            for (index iDomain = 0; iDomain < domains; ++iDomain)
                if (simulation.Local(iDomain))
                    action(iDomain, simulation.GetDomain(iDomain));
        }

        void ReportAllocations(const std::string &token)
        {
            // Explicit host output boundary. Rank-local files require no new
            // agreement/gather and do not interrupt a queued numerical region.
            const std::filesystem::path prefix(output.prefix);
            const auto name = (prefix.parent_path() / "memory" /
                               (prefix.filename().string() + "-memory-R" + std::to_string(mpi.rank) + token + ".txt"))
                                  .string();
            std::ofstream file(name);
            DNDS_check_throw_info(bool(file), "cannot open allocation report: " + name);
            Runtime::AllocationDiagnostics::Snapshot().Dump(file);
            file << "Untracked provider allocations: AmgX/NCCL/internal pools (unknown bytes)\n";
            file.flush();
            DNDS_check_throw_info(bool(file), "failed to write allocation report: " + name);
        }
        void MaybeReportAllocations()
        {
            if (output.diagnostics.allocations && !allocationReportWritten &&
                (output.diagnostics.allocationWarmupSteps < 0 ||
                 simulation.Step() >= uint64_t(output.diagnostics.allocationWarmupSteps)))
            {
                ReportAllocations(OutputPlan::StepToken(simulation.Step()));
                allocationReportWritten = true;
            }
        }

        void InitializeFiles()
        {
            MPI::LocalStage(mpi, [&]
                            {
                if (mpi.rank == 0)
                {
                    const auto parent =
                        std::filesystem::path(output.prefix).parent_path();
                    if (!parent.empty())
                        std::filesystem::create_directories(parent);
                    if (output.diagnostics.allocations)
                        std::filesystem::create_directories(parent / "memory");
                    std::ofstream resolved(plan.ResolvedPath());
                    resolved << currentConfigJson.dump(2) << '\n';
                    DNDS_check_throw_info(
                        bool(resolved),
                        "cannot write resolved Cauchy configuration");
                } });
            Visit([&](index iDomain, Domain &domain)
                  { MPI::LocalStage(
                        MPIInfo(domain.Discretization().Metrics().Communicator()),
                        [&]
                        {
                            std::function<void(const CsvRow &)> retain;
                            if (control)
                                retain = [this, iDomain](const CsvRow &row)
                                { control->RetainHistory(iDomain, row); };
                            std::vector<BoundaryFluxSinkConfig> bndSinks;
                            bndSinks.reserve(boundarySpecs[size_t(iDomain)].size());
                            for (const auto &spec : boundarySpecs[size_t(iDomain)])
                            {
                                std::optional<CsvLogPosition> bndResume;
                                if (continueOutput && size_t(iDomain) < state.boundaryHistory.size())
                                {
                                    const auto it = state.boundaryHistory[size_t(iDomain)].find(spec.canonicalName);
                                    if (it != state.boundaryHistory[size_t(iDomain)].end())
                                        bndResume = it->second;
                                }
                                std::function<void(int, const CsvRow &)> bndRetain;
                                if (control)
                                    bndRetain = [this, iDomain](int zone, const CsvRow &row)
                                    { control->UpdateBoundaryLatest(iDomain, zone, CsvRowToJson(row)); };
                                bndSinks.push_back({spec,
                                                    plan.BoundaryIntegrationBare(iDomain, spec.canonicalName),
                                                    bndResume,
                                                    std::move(bndRetain)});
                            }
                            progress[size_t(iDomain)] =
                                std::make_unique<ProgressOutputController>(
                                    domain.Equation().Schema(), output,
                                    domains == 1 ? std::string{} : domain.Name(), plan.HistoryBare(iDomain),
                                    simulation.Placement().DomainRank(iDomain) == 0,
                                    logger,
                                    continueOutput
                                        ? std::optional(state.history.at(
                                              size_t(iDomain)))
                                        : std::nullopt,
                                    std::move(retain),
                                    std::move(bndSinks));
                        }); });
            Json localDomains = Json::array();
            for (index iDomain = 0; iDomain < domains; ++iDomain)
                if (simulation.Local(iDomain) &&
                    simulation.Placement().DomainRank(iDomain) == 0)
                {
                    const auto &domain = simulation.GetDomain(iDomain);
                    Json bndList = Json::array();
                    for (const auto &spec : boundarySpecs[size_t(iDomain)])
                        bndList.push_back({{"zone", spec.zone},
                                           {"rawName", spec.rawName},
                                           {"canonicalName", spec.canonicalName},
                                           {"csv", plan.BoundaryIntegrationBare(iDomain, spec.canonicalName) + ".csv"},
                                           {"sign", "negative_outward_rhs_flux"}});
                    localDomains.push_back(
                        {{"index", iDomain},
                         {"name", domain.Name()},
                         {"history", plan.HistoryBare(iDomain) + ".csv"},
                         {"eventLog", output.eventLog ? Json(output.prefix + "-events-rank" + std::to_string(mpi.rank) + ".log") : Json()},
                         {"components", EquationComponentNames(
                                            domain.Equation().Schema())},
                         {"boundaryIntegrations", std::move(bndList)}});
                }
            const auto manifestDomains = GatherJsonArrays(mpi, localDomains);
            MPI::LocalStage(mpi, [&]
                            {
                if (mpi.rank == 0)
                {
                    if (control)
                        for (const auto &dom : manifestDomains)
                            control->SetBoundaryMetadata(dom.at("index").get<index>(),
                                                         dom.value("boundaryIntegrations", Json::array()));
                    const Json manifest{
                        {"schema", "dndsr.cauchy.run-output.v1"},
                        {"resolvedConfig", plan.ResolvedPath()},
                        {"historySchema", output.historyStateBounds ? "dndsr.cauchy.detailed-history.v3" : "dndsr.cauchy.detailed-history.v2"},
                        {"progressStatus",
                         {{"nonlinearIteration", 0},
                          {"implicitStageEnd", 1},
                          {"acceptedStepEnd", 2}}},
                        {"domains", manifestDomains}};
                    std::ofstream file(plan.RunManifestPath());
                    file << manifest.dump(2) << '\n';
                    DNDS_check_throw_info(bool(file),
                                          "cannot write Cauchy run manifest");
                } });
        }

        Json CaptureState()
        {
            std::vector<int64_t> positions(size_t(2 * domains), 0);
            std::vector<int> roots(size_t(domains), 0);
            Json localBndPositions = Json::array();
            Visit([&](index iDomain, Domain &)
                  {
                      progress[size_t(iDomain)]->Flush();
                      if (simulation.Placement().DomainRank(iDomain) == 0)
                      {
                          const auto cursor = progress[size_t(iDomain)]->HistoryPosition();
                          positions[size_t(2 * iDomain)] = cursor.block;
                          positions[size_t(2 * iDomain + 1)] = cursor.line;
                          roots[size_t(iDomain)] = 1;
                          const auto bndPos = progress[size_t(iDomain)]->BoundaryHistoryPositions();
                          if (!bndPos.empty())
                          {
                              Json obj = Json::object();
                              for (const auto &[name, pos] : bndPos)
                                  obj[name] = {{"block", pos.block}, {"rows", pos.line}};
                              localBndPositions.push_back({{"domain", iDomain}, {"positions", std::move(obj)}});
                          }
                      } });
            MPI::AllreduceInPlace<MPI::Reduction::Sum>(
                MPI::ContiguousView<int64_t>{positions.data(), positions.size()},
                mpi.comm);
            MPI::AllreduceInPlace<MPI::Reduction::Sum>(
                MPI::ContiguousView<int>{roots.data(), roots.size()}, mpi.comm);
            const auto gatheredBnd = GatherJsonArrays(mpi, localBndPositions);
            const auto allBnd = BroadcastJson(mpi, gatheredBnd);
            state.checkpointInterval = output.checkpointInterval;
            state.visualizationInterval = plan.VisualizationInterval();
            state.checkpointTimeInterval = output.checkpointTimeInterval;
            state.visualizationTimeInterval = plan.VisualizationTimeInterval();
            state.nextCheckpointTime = nextCheckpointTime;
            state.nextVisualizationTime = nextVisualizationTime;
            state.boundaryHistory.assign(size_t(domains), {});
            for (const auto &entry : allBnd)
            {
                const auto d = entry.at("domain").get<index>();
                for (const auto &[name, cur] : entry.at("positions").items())
                    state.boundaryHistory[size_t(d)][name] = {cur.at("block").get<int>(), cur.at("rows").get<int64_t>()};
            }
            for (index iDomain = 0; iDomain < domains; ++iDomain)
            {
                DNDS_check_throw_info(
                    roots[size_t(iDomain)] == 1 &&
                        positions[size_t(2 * iDomain)] >= 0 &&
                        positions[size_t(2 * iDomain + 1)] >= 0,
                    "Cauchy output history cursor has no unique domain root");
                state.history[size_t(iDomain)] = {int(positions[size_t(2 * iDomain)]), positions[size_t(2 * iDomain + 1)]};
            }
            return state.ToJson();
        }

        Json CaptureDomainState(index iDomain)
        {
            auto local = state;
            auto &domain = simulation.GetDomain(iDomain);
            const MPIInfo domainMPI(
                domain.Discretization().Metrics().Communicator());
            CsvLogPosition cursor;
            Json bndObj = Json::object();
            MPI::LocalStage(domainMPI, [&]
                            {
                progress[size_t(iDomain)]->Flush();
                if (domainMPI.rank == 0)
                {
                    cursor = progress[size_t(iDomain)]->HistoryPosition();
                    for (const auto &[name, pos] : progress[size_t(iDomain)]->BoundaryHistoryPositions())
                        bndObj[name] = {{"block", pos.block}, {"rows", pos.line}};
                } });
            int64_t encoded[2]{cursor.block, cursor.line};
            MPI::Bcast(encoded, 2, MPI_INT64_T, 0, domainMPI.comm);
            const auto agreedBnd = BroadcastJson(domainMPI, bndObj);
            local.checkpointInterval = output.checkpointInterval;
            local.visualizationInterval = plan.VisualizationInterval();
            local.checkpointTimeInterval = output.checkpointTimeInterval;
            local.visualizationTimeInterval = plan.VisualizationTimeInterval();
            local.history[size_t(iDomain)] = {int(encoded[0]), encoded[1]};
            if (local.boundaryHistory.size() < size_t(domains))
                local.boundaryHistory.resize(size_t(domains));
            local.boundaryHistory[size_t(iDomain)].clear();
            for (const auto &[name, cur] : agreedBnd.items())
                local.boundaryHistory[size_t(iDomain)][name] = {cur.at("block").get<int>(), cur.at("rows").get<int64_t>()};
            local.nextCheckpointTime = nextCheckpointTime;
            local.nextVisualizationTime = nextVisualizationTime;
            return local.ToJson();
        }

        void InstallObservers()
        {
            Visit([&](index iDomain, Domain &domain)
                  {
                      domain.SetProgressObserver(
                          [this, iDomain](const NonlinearProgress &entry)
                          {
                              lastObservedStage = entry.stage;
                              lastObservedIteration = entry.iteration;
                              progress[size_t(iDomain)]->CaptureIteration(entry, progressOffset + std::chrono::duration<real>(std::chrono::steady_clock::now() - progressStart).count());
                              PublishStatus("running");
                          },
                          output.historyStateBounds ? output.implicit.historyInterval : 0,
                          output.implicit.consoleInterval > 0,
                          boundaryZones[size_t(iDomain)],
                          boundaryZones[size_t(iDomain)].empty() ? 0 : output.implicit.historyInterval);
                      domain.SetQuiescentObserver(
                          [this, iDomain](const NonlinearSolveState &nonlinear)
                          { InternalSnapshot(iDomain, nonlinear); }); });
        }

        void InternalSnapshot(index iDomain,
                              const NonlinearSolveState &nonlinear)
        {
            const bool ditr = time.method == "ditr";
            const index completedIteration =
                ditr || nonlinear.terminal ? nonlinear.iteration
                                           : nonlinear.iteration + 1;
            const auto step = simulation.Step() + 1;
            const auto selected = plan.SelectedSnapshots(step, nonlinear.stage, completedIteration);
            const auto intervalDue = [completedIteration](index interval)
            { return completedIteration > 0 && interval > 0 && completedIteration % interval == 0; };
            const bool liveSnap = pendingInternalSnapshot.has_value();
            const bool liveCkpt = liveSnap && pendingInternalSnapshot->checkpoint;
            const bool liveVis = liveSnap && pendingInternalSnapshot->visualization;
            const uint64_t liveId = liveSnap ? pendingInternalSnapshot->id : 0;
            const bool checkpoint = iterationStop || liveCkpt || selected.checkpoint || intervalDue(internalCheckpointInterval);
            const bool visualization = !iterationStop && (liveVis || selected.visualization || intervalDue(internalVisualizationInterval));
            // Emit progress at the existing quiescent nonlinear boundary
            // before staging any output or entering file I/O.
            progress[size_t(iDomain)]->FlushIterations(checkpoint || visualization);
            if (output.diagnostics.allocations && !allocationReportWritten &&
                output.diagnostics.allocationWarmupSteps < 0 && completedIteration > 0)
            {
                ReportAllocations(OutputPlan::InternalToken(step, nonlinear.stage, completedIteration));
                allocationReportWritten = true;
            }
            if (!checkpoint && !visualization)
                return;
            const bool writer = simulation.Placement().DomainRank(iDomain) == 0;
            const auto token = OutputPlan::InternalToken(step, nonlinear.stage, completedIteration) +
                               (iterationStop ? "-stop" : (liveSnap ? "-live" + std::to_string(liveId) : ""));
            if (iterationStop)
                stopCheckpoint = plan.InternalCheckpointPath(iDomain, token);
            if (writer)
                logger->Info(fmt::format("Writing {} D{}{}",
                                         checkpoint && visualization ? "checkpoint and visualization"
                                         : checkpoint                ? "checkpoint"
                                                                     : "visualization",
                                         iDomain, token));
            auto &domain = simulation.GetDomain(iDomain);
            const auto prefix = plan.VisualizationPrefix(iDomain, token);
            Json artifacts = Json::array();
            if (checkpoint)
                artifacts.push_back(
                    {{"kind", "checkpoint"},
                     {"path", plan.InternalCheckpointPath(iDomain, token)},
                     {"schema", "dndsr.cauchy.checkpoint.internal.v1"}});
            if (visualization)
            {
                artifacts.push_back(
                    {{"kind", "volume"},
                     {"path", prefix + ".vtkhdf"},
                     {"schema", "dndsr.cauchy.visualization.v1"}});
                artifacts.push_back(BoundaryArtifact(prefix, output.boundaryFacialFluxes,
                                                     domain.InternalObservationTime()));
            }
            const Json manifest{
                {"schema", "dndsr.cauchy.internal-snapshot.v1"},
                {"restartScope",
                 checkpoint ? "exact_internal" : "visualization_only"},
                {"domain", iDomain},
                {"step", simulation.Step() + 1},
                {"stage", nonlinear.stage},
                {"coupledStage", nonlinear.coupledStage},
                {"iteration", completedIteration},
                {"time", domain.InternalObservationTime()},
                {"artifacts", std::move(artifacts)}};
            auto controller = checkpoint ? CaptureDomainState(iDomain) : Json();
            if (liveSnap && iDomain == domains - 1)
                pendingInternalSnapshot.reset();
            if (asynchronous)
            {
                asynchronous->SubmitInternal(simulation, config, plan, iDomain, token, nonlinear,
                                             checkpoint, visualization, std::move(controller), manifest);
                if (liveSnap && control && mpi.rank == 0)
                    control->RecordAppliedSnapshot(liveId, step, nonlinear.stage, completedIteration);
                return;
            }
            if (visualization)
                domain.WriteInternalVisualization(prefix, output.hdfDeflateLevel);
            if (checkpoint)
                domain.WriteInternalCheckpoint(plan.InternalCheckpointPath(iDomain, token), currentConfigJson, nonlinear, controller);
            if (liveSnap && control && mpi.rank == 0)
                control->RecordAppliedSnapshot(liveId, step, nonlinear.stage, completedIteration);
            if (!writer)
                return;
            std::ofstream file(plan.InternalSnapshotManifestPath(iDomain, token));
            file << manifest.dump(2) << '\n';
            DNDS_check_throw_info(bool(file),
                                  "cannot write Cauchy internal snapshot manifest");
        }

        void FlushPending()
        {
            Visit([&](index iDomain, Domain &domain)
                  { MPI::LocalStage(
                        MPIInfo(domain.Discretization().Metrics().Communicator()),
                        [&]
                        { progress[size_t(iDomain)]->FlushIterations(); }); });
        }

        bool Record(real dt, index retries,
                    const std::vector<PhysicalStepLimit> &limits,
                    bool useLimits, bool printConsole, bool forceHistory)
        {
            int allReached = 1;
            Visit([&](index iDomain, Domain &domain)
                  {
                      const auto values = domain.Diagnostics(
                          config.Domains()[iDomain].settings, output.forceZones);
                      const bool writeStepHistory =
                          forceHistory ||
                          (output.historyInterval > 0 && domain.Epoch().acceptedStep % uint64_t(output.historyInterval) == 0);
                      std::vector<real> stateBounds;
                      if (output.historyStateBounds && writeStepHistory)
                          stateBounds = domain.MeasureStateBounds(false);
                      std::vector<BoundaryFluxIntegral> bndFluxes;
                      if (writeStepHistory && !boundaryZones[size_t(iDomain)].empty() &&
                          domain.LastImplicitStages().empty())
                          bndFluxes = domain.MeasureBoundaryFluxes(boundaryZones[size_t(iDomain)]);
                      auto &baseline = baselines[size_t(iDomain)];
                      if (baseline == 0 && std::isfinite(values[4]))
                          baseline = std::max(values[4], real(1e-300));
                      const auto limit = useLimits
                                             ? limits.at(size_t(iDomain))
                                             : PhysicalStepLimit{};
                      index nonlinearIterations = 0, linearIterations = 0;
                      for (const auto &report : domain.LastImplicitStages())
                      {
                          nonlinearIterations += report.iterations;
                          linearIterations += report.linearIterations;
                      }
                      MPI::LocalStage(
                          MPIInfo(domain.Discretization().Metrics().Communicator()),
                          [&]
                          {
                              const ProgressLocation location{
                                  domain.Epoch().acceptedStep, -1, -1, -1,
                                  retries, domain.Time(), dt,
                                  MPI_Wtime() - started, time.steady};
                              for (index iStage = 0;
                                   iStage < index(domain.LastImplicitStages().size());
                                   ++iStage)
                              {
                                  auto stage = location;
                                  stage.stage = iStage;
                                  progress[size_t(iDomain)]->StageEnd(
                                      stage,
                                      domain.LastImplicitStages()[size_t(iStage)]);
                              }
                              AcceptedStepProgress accepted;
                              accepted.diagnostics = values;
                              accepted.stateBounds = std::move(stateBounds);
                              accepted.boundaryFluxes = std::move(bndFluxes);
                              accepted.faceFallbacks = domain.LastFaceFallbacks();
                              accepted.retries = retries;
                              accepted.implicitStages =
                                  index(domain.LastImplicitStages().size());
                              accepted.cappedStages = index(std::count_if(
                                  domain.LastImplicitStages().begin(), domain.LastImplicitStages().end(),
                                  [](const auto &report) { return !report.converged; }));
                              accepted.nonlinearIterations = nonlinearIterations;
                              accepted.linearIterations = linearIterations;
                              accepted.stepWallSeconds = stepWallSeconds;
                              accepted.coupledIterations =
                                  domain.LastCoupledIterations();
                              accepted.coupledResidualTargetRatio =
                                  domain.LastCoupledTargetRatio();
                              if (!domain.LastImplicitStages().empty())
                              {
                                  accepted.residual = domain.LastImplicitStages().back().residual;
                                  accepted.pseudoCFL = domain.LastImplicitStages().back().pseudoCFL;
                              }
                              accepted.positivity = domain.LastPositivity();
                              accepted.timePPLimitedCells = limit.limitedCells;
                              accepted.timePPMinimumAlpha = limit.minimumAlpha;
                              accepted.timeGrowthLimited = limit.growthLimited;
                              progress[size_t(iDomain)]->StepEnd(
                                  location, accepted, forceHistory, printConsole);
                          });
                      const bool outerTarget =
                          time.steadyTolerance <= 0 ||
                          values[4] / baseline <= time.steadyTolerance;
                      if (time.steady)
                          allReached &=
                              !domain.LastImplicitStages().empty() &&
                              std::all_of(
                                  domain.LastImplicitStages().begin(),
                                  domain.LastImplicitStages().end(),
                                  [](const auto &stage)
                                  { return stage.converged; }) &&
                              outerTarget;
                      else
                          allReached &=
                              time.steadyTolerance > 0 &&
                              domain.Epoch().acceptedStep >=
                                  uint64_t(time.steadyMinSteps) &&
                              outerTarget; });
            allReached = MPI::AllreduceValue<MPI::Reduction::Min>(allReached,
                                                                  mpi.comm);
            lastObservedStage = -1;
            lastObservedIteration = -1;
            PublishStatus("running");
            return bool(allReached);
        }

        void WriteSnapshot(const std::string &token, bool checkpoint,
                           bool visualization)
        {
            const auto manifest = SnapshotManifest(token, checkpoint, visualization);
            if (asynchronous)
            {
                asynchronous->Submit(simulation, config, plan, token, checkpoint, visualization,
                                     checkpoint ? CaptureState() : Json(), manifest);
                return;
            }
            if (checkpoint)
                simulation.WriteCheckpoint(plan.CheckpointPath(token),
                                           output.checkpointReconstruction,
                                           CaptureState());
            if (visualization)
                Visit([&](index iDomain, Domain &domain)
                      { domain.WriteVisualization(
                            plan.VisualizationPrefix(iDomain, token),
                            output.hdfDeflateLevel); });
            MPI::LocalStage(mpi, [&]
                            {
                if (mpi.rank != 0)
                    return;
                std::ofstream file(plan.SnapshotManifestPath(token));
                file << manifest.dump(2) << '\n';
                file.close();
                DNDS_check_throw_info(bool(file),
                                      "cannot write Cauchy snapshot manifest"); });
        }

        Json SnapshotManifest(const std::string &token, bool checkpoint, bool visualization) const
        {
            Json artifacts = Json::array();
            if (checkpoint)
                artifacts.push_back(
                    {{"kind", "checkpoint"},
                     {"path", plan.CheckpointPath(token)},
                     {"schema", "dndsr.cauchy.checkpoint.v6"}});
            if (visualization)
                for (index iDomain = 0; iDomain < domains; ++iDomain)
                {
                    const auto prefix =
                        plan.VisualizationPrefix(iDomain, token);
                    artifacts.push_back(
                        {{"kind", "volume"},
                         {"domain", iDomain},
                         {"path", prefix + ".vtkhdf"},
                         {"schema", "dndsr.cauchy.visualization.v1"}});
                    auto boundary = BoundaryArtifact(prefix, output.boundaryFacialFluxes, simulation.Time());
                    boundary["domain"] = iDomain;
                    artifacts.push_back(std::move(boundary));
                }
            return Json{{"schema", "dndsr.cauchy.snapshot.v1"},
                        {"step", simulation.Step()},
                        {"time", simulation.Time()},
                        {"token", token},
                        {"artifacts", std::move(artifacts)}};
        }

        bool CompleteAcceptedStep(
            real dt, index retries,
            const std::vector<PhysicalStepLimit> &limits, bool useLimits,
            bool printConsole)
        {
            // Read the existing host boundary only; no new device fence or
            // timing collective. Include retries, exclude following file I/O.
            stepWallSeconds = stepStarted ? MPI_Wtime() - *stepStarted : 0;
            stepStarted.reset();
            bool checkpointTimeDue = false, visualizationTimeDue = false;
            if (!time.steady && output.checkpointTimeInterval > 0)
                while (OutputPlan::TimeAt(nextCheckpointTime,
                                          output.checkpointTimeInterval) <=
                       simulation.Time())
                {
                    checkpointTimeDue = true;
                    ++nextCheckpointTime;
                }
            if (!time.steady && plan.VisualizationTimeInterval() > 0)
                while (OutputPlan::TimeAt(nextVisualizationTime,
                                          plan.VisualizationTimeInterval()) <=
                       simulation.Time())
                {
                    visualizationTimeDue = true;
                    ++nextVisualizationTime;
                }
            const bool checkpointStepDue = output.checkpointInterval > 0 &&
                                           simulation.Step() %
                                                   uint64_t(output.checkpointInterval) ==
                                               0;
            const bool visualizationStepDue =
                plan.VisualizationInterval() > 0 &&
                simulation.Step() % uint64_t(plan.VisualizationInterval()) == 0;
            const auto selected = plan.SelectedSnapshots(simulation.Step());
            const bool checkpoint = checkpointTimeDue || checkpointStepDue || selected.checkpoint;
            const bool visualization =
                visualizationTimeDue || visualizationStepDue || selected.visualization;
            const bool reached =
                Record(dt, retries, limits, useLimits,
                       printConsole || (output.consoleInterval > 0 && (checkpoint || visualization)),
                       false);
            MaybeReportAllocations();
            if (checkpoint || visualization)
            {
                const auto token = checkpointTimeDue || visualizationTimeDue
                                       ? OutputPlan::TimeToken(simulation.Time())
                                       : OutputPlan::StepToken(simulation.Step());
                WriteSnapshot(token, checkpoint, visualization);
            }
            return reached;
        }

        int Finalize(RunStop stop, bool reached)
        {
            DNDS_check_throw_info(!finalized,
                                  "Cauchy output coordinator finalized twice");
            finalized = true;
            if (stop == RunStop::Requested)
            {
                if (!iterationStop)
                {
                    const auto token = OutputPlan::StepToken(simulation.Step()) + "-stop";
                    stopCheckpoint = plan.CheckpointPath(token);
                    WriteSnapshot(token, true, false);
                }
                DNDS_check_throw_info(!stopCheckpoint.empty(), "Cauchy stop did not publish a checkpoint");
            }
            else if (output.finalSnapshot)
                WriteSnapshot("-final", true, true);
            if (asynchronous)
                asynchronous->Drain();
            if (output.diagnostics.allocations)
                ReportAllocations("-final");
            Visit([&](index iDomain, Domain &)
                  { progress[size_t(iDomain)]->Flush(); });
            logger->Flush();
            const bool timeReached =
                time.tEnd <= 0 || EquationTimeStepper::ReachedTimeTarget(
                                      simulation.Time(), time.tEnd);
            const bool wallStop = stop == RunStop::WallLimit;
            const bool stepStop =
                stop == RunStop::None &&
                ((!timeReached && time.tEnd > 0) ||
                 ((time.steady || time.steadyTolerance > 0) && !reached));
            const std::string statusText =
                stop == RunStop::Requested ? "graceful-stop"
                : wallStop                 ? "wall-capped"
                : stepStop                 ? "step-capped"
                : reached                  ? "steady-converged"
                                           : "target-reached";
            PublishStatus("finished", {{"finalStatus", statusText},
                                       {"checkpoint", stopCheckpoint}});
            MPI::LocalStage(mpi, [&]
                            {
                if (mpi.rank == 0)
                {
                    Json result{
                        {"time", simulation.Time()},
                        {"steps", simulation.Step()},
                        {"wallSeconds", MPI_Wtime() - started},
                        {"status", statusText},
                        {"steadyTargetMet", reached},
                        {"steadyMode", time.steady},
                        {"physicalTimeTarget", time.tEnd},
                        {"steadyTolerance", time.steadyTolerance}};
                    if (stop == RunStop::Requested)
                    {
                        result["stopBoundary"] = iterationStop ? "iteration" : "step";
                        result["checkpoint"] = stopCheckpoint;
                        logger->Info("Graceful stop checkpoint complete: " + stopCheckpoint);
                    }
                    std::ofstream summary(plan.SummaryPath());
                    summary << result.dump(2) << '\n';
                    DNDS_check_throw_info(bool(summary),
                                          "cannot write Cauchy summary");
                } });
            return wallStop || stepStop ? 2 : 0;
        }
    };

    OutputCoordinator::OutputCoordinator(Simulation &simulation,
                                         const SimulationConfig &config,
                                         MPIInfo mpi, real started, RunControl *control,
                                         std::function<void()> pollInput)
        : _data(std::make_unique<Data>(simulation, config, mpi, started, control, std::move(pollInput))) {}
    OutputCoordinator::~OutputCoordinator() = default;

    const TimeSettings &OutputCoordinator::LiveTime() const { return _data->time; }
    const OutputSettings &OutputCoordinator::LiveOutput() const { return _data->output; }
    std::string OutputCoordinator::RunManifestPath() const { return _data->plan.RunManifestPath(); }
    std::shared_ptr<OperationalLogger> OutputCoordinator::Logger() const { return _data->logger; }

    OutputPlan::StepTarget OutputCoordinator::SnapStep(real proposedDt) const
    {
        return _data->plan.SnapStep(
            _data->simulation.Time(), proposedDt, _data->time.tEnd,
            _data->nextCheckpointTime, _data->nextVisualizationTime);
    }
    void OutputCoordinator::BeginAttempt(real dt, index retries,
                                         real targetTime)
    {
        if (!_data->stepStarted)
            _data->stepStarted = MPI_Wtime();
        _data->Visit([&](index iDomain, Domain &)
                     { _data->progress[size_t(iDomain)]->BeginAttempt(
                           {_data->simulation.Step() + 1, -1, -1, -1, retries, targetTime,
                            dt, MPI_Wtime() - _data->started, _data->time.steady}); });
    }
    void OutputCoordinator::FlushPending() { _data->FlushPending(); }
    void OutputCoordinator::WarnRetry(const std::string &reason)
    {
        _data->logger->Warn("Retry candidate with half dt: " + reason);
    }
    bool OutputCoordinator::CompleteAcceptedStep(
        real dt, index retries, const std::vector<PhysicalStepLimit> &limits,
        bool useLimits, bool printConsole)
    {
        return _data->CompleteAcceptedStep(dt, retries, limits, useLimits,
                                           printConsole);
    }
    RunStop OutputCoordinator::PollStop(bool wallLimitReached)
    {
        return _data->PollStop(wallLimitReached);
    }
    int OutputCoordinator::Finalize(RunStop stop, bool reached)
    {
        return _data->Finalize(stop, reached);
    }
}
