#include "Cauchy/State/CPU/FieldKernels.hpp"
#include "Cauchy/Implicit/LinearSolver.hpp"
#include "Cauchy/Core/OperatorSchedule.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Cauchy/State/EquationDofCommunication.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "Cauchy/Core/LocalStage.hpp"
#include "DNDS/MPI/Failure.hpp"
#include "DNDS/Runtime/Execution/SessionAccess.hpp"
#include <array>
#include <optional>
#include <utility>
#include <cstdlib>
#include <chrono>
#include <cstdio>
#include <iostream>
#include "DNDS/Serializer/JsonUtil.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    namespace A = Algebra;
    namespace
    {
        /// Opt-in controller diagnosis. Local clocks only, no extra MPI or
        /// numeric-policy changes. Counters are per solver instance; disabled
        /// runs do not query a clock. Nested total includes all named phases.
        struct LinearProfile
        {
            enum Phase
            {
                Factor,
                Action,
                Triangular,
                SGS,
                Products,
                Algebra,
                Total,
                PhaseNumber
            };
            bool enabled = std::getenv("DNDS_CAUCHY_PROFILE_LINEAR") != nullptr;
            bool allRanks = std::getenv("DNDS_CAUCHY_PROFILE_ALL_RANKS") != nullptr;
            std::array<real, PhaseNumber> seconds{};
            std::array<index, PhaseNumber> calls{};
            index instance = 0, solves = 0;
            LinearProfile()
            {
                if (enabled)
                {
                    static thread_local index next = 0;
                    instance = next++;
                }
            }
            class Scope
            {
                using Clock = std::chrono::steady_clock;
                LinearProfile *_profile;
                Phase _phase;
                Clock::time_point _start{};

            public:
                Scope(LinearProfile &profile, Phase phase) : _profile(profile.enabled ? &profile : nullptr), _phase(phase)
                {
                    if (_profile)
                        _start = Clock::now();
                }
                ~Scope()
                {
                    if (_profile)
                    {
                        _profile->seconds[_phase] += std::chrono::duration<real>(Clock::now() - _start).count();
                        ++_profile->calls[_phase];
                    }
                }
            };
            void Record(int rank)
            {
                if (!enabled)
                    return;
                ++solves;
                if ((!rank || allRanks) && solves % 32 == 0)
                {
                    const auto record = nlohmann::ordered_json{{"rank", rank}, {"solver_instance", instance}, {"solves", solves}, {"phases", {"factor", "action", "triangular", "sgs", "products", "algebra", "total"}}, {"seconds", seconds}, {"calls", calls}}.dump();
                    std::fprintf(stdout, "CAUCHY_LINEAR_PROFILE %s\n", record.c_str());
                }
            }
        };
        bool UsesILU(const LinearSolveSettings &settings)
        {
            return settings.method == "ilu" || (settings.method == "gmres" && settings.preconditioner == "ilu");
        }
        struct FieldLinearOperations : Linear::FieldKrylovOperations
        {
            using State = Field;
            using Iteration = LinearProblemContext::SGSIteration;
            static index Width(const State &state) { return index(state.Width()); }
            static State Alias(const State &state) { return state; }
            static bool SameStorage(const State &a, const State &b)
            {
                return a.Owned().SharedStorage() == b.Owned().SharedStorage() &&
                       a.Ghost().SharedStorage() == b.Ghost().SharedStorage();
            }
            static void Agree(const State &, MPI_Comm) {}
            static bool LocalFinite(const State &state)
            {
                const auto read = state.Owned().Read();
                return CPU::FiniteRows(read.View<R::UniformTableRows>(), Width(state));
            }
            template <class S, class F>
            static void VisitFields(S &state, F &&visit) { visit(state); }
            static std::vector<real> LocalProducts(const State &x, const std::vector<const State *> &probes,
                                                   const std::vector<real> &inverseScales)
            {
                std::vector<real> local(probes.size(), 0);
                const auto read = x.Owned().Read();
                const auto source = read.View<R::UniformTableRows>();
                std::vector<R::Array<real>::ReadLease> retained;
                std::vector<R::ArrayView<const real, R::AccessBoundsChecks, R::UniformTableRows>> values;
                retained.reserve(probes.size());
                values.reserve(probes.size());
                for (const auto *probe : probes)
                {
                    retained.push_back(probe->Owned().Read());
                    values.push_back(retained.back().View<R::UniformTableRows>());
                }
                Algebra::Detail::AccumulateLocalProducts(source, values, source.RowCount(), Width(x), local, inverseScales);
                return local;
            }
        };
        struct DofLinearOperations : EquationDofOperations
        {
            using Iteration = LinearProblemContext::DofSGSIteration;
            static index Width(const State &state) { return state.ComponentNumber(); }
            static State Alias(const State &state) { return state.NumericAlias(); }
            static bool SameStorage(const State &a, const State &b)
            {
                if (a.PhysicalFieldNumber() != b.PhysicalFieldNumber())
                    return false;
                for (index iField = 0; iField < a.PhysicalFieldNumber(); ++iField)
                    if (!FieldLinearOperations::SameStorage(a.PhysicalField(iField), b.PhysicalField(iField)))
                        return false;
                return true;
            }
            static void Agree(const State &state, MPI_Comm communicator) { AgreeEquationBinding(state, communicator); }
            static bool LocalFinite(const State &state)
            {
                const auto read = state.ReadOwned();
                const auto layout = read.ComponentLayout();
                return CPU::FiniteRows(layout.View(), Width(state));
            }
            static std::vector<real> LocalProducts(const State &x, const std::vector<const State *> &probes,
                                                   const std::vector<real> &inverseScales)
            {
                return EquationDofOperations::LocalProducts(x, probes, inverseScales);
            }
            template <class S, class F>
            static void VisitFields(S &state, F &&visit)
            {
                for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                    visit(state.PhysicalField(iField));
            }
        };
        template <class Operations>
        struct StateLinearSolverData
        {
            using Field = typename Operations::State;
            using Iteration = typename Operations::Iteration;
            using GMRES = Linear::StateGMRES<Field, Operations>;
            using Ops = Operations;
            std::shared_ptr<LinearProblemContext> problem;
            MPIInfo mpi;
            LinearSolveSettings settings;
            LinearExecution execution;
            std::unique_ptr<R::EagerEngine> standaloneEngine;
            R::ExecutionStream compute, communication;
            std::unique_ptr<GMRES> gmres;
            std::optional<Field> solution;
            Field old, next, forward, action, residual;
            struct Input
            {
                Field field; // Retained allocation/shape handles, not a facade address.
                MPI_Comm comm = MPI_COMM_NULL;
                struct Slot
                {
                    R::TransferBinding binding;
                    std::optional<R::TransferOperation> operation;
                };
                std::vector<Slot> slots;
                bool active = false;
            };
            std::vector<Input> inputs;
            /// Slots are retained at binding time. Await imports each managed
            /// completion into the compute lane; dropping the public handles
            /// does not block or release active payload owners. The enclosing
            /// operator schedule drains before returning. A partially posted
            /// launch failure aborts, never attempts distributed recovery.
            class Exchange
            {
                OptionalRef<Input> _input;
                void Retire() noexcept
                {
                    if (!_input)
                        return;
                    for (auto &slot : _input->slots)
                        slot.operation.reset();
                    _input->active = false;
                    _input.Reset();
                }

            public:
                Exchange(Input &input, const OperatorSchedule &schedule) : _input(input)
                {
                    DNDS_check_throw_info(!input.active, "linear halo is already active");
                    input.active = true;
                    try
                    {
                        for (auto &slot : input.slots)
                            slot.operation.emplace(schedule.Begin(slot.binding));
                    }
                    catch (...)
                    {
                        MPI::AbortIfSolverFailure("Cauchy linear operation failed");
                        // A first binding's coordinated admission rejection has
                        // no active payload. Never wait on earlier fields when
                        // another binding fails after this exchange has posted.
                        for (const auto &slot : input.slots)
                            if (slot.operation)
                                MPI::Abort(input.comm, "linear halo launch failed after an earlier field posted");
                        input.active = false;
                        throw;
                    }
                }
                Exchange(const Exchange &) = delete;
                Exchange &operator=(const Exchange &) = delete;
                Exchange(Exchange &&other) noexcept : _input(std::exchange(other._input, OptionalRef<Input>{})) {}
                ~Exchange() { Retire(); }
                void Await(const OperatorSchedule &schedule)
                {
                    if (!_input)
                        return;
                    for (auto &slot : _input->slots)
                        schedule.Await(*slot.operation);
                    Retire();
                }
            };
            bool failed = false;
            std::vector<real> inverseComponentScales;
            mutable LinearProfile profile;
            StateLinearSolverData(std::shared_ptr<LinearProblemContext> matrix, const Field &prototype, LinearSolveSettings controls, LinearExecution policy)
                : problem(std::move(matrix)), mpi(problem->Space().Communicator()), settings(std::move(controls)), execution(policy),
                  old(Ops::NewVector(prototype)), next(Ops::NewVector(prototype)), forward(Ops::NewVector(prototype)), action(Ops::NewVector(prototype)), residual(Ops::NewVector(prototype))
            {
                if (execution.streams)
                {
                    compute = execution.streams->compute;
                    communication = execution.streams->communication;
                    auto engine = compute.LockEngine();
                    DNDS_check_throw_info(engine.Owns(communication) &&
                                              compute.Context().device == R::Device::Host() &&
                                              communication.Context().device == R::Device::Host(),
                                          "linear execution has foreign lanes or an unavailable backend");
                }
                else
                {
                    standaloneEngine = std::make_unique<R::EagerEngine>();
                    compute = standaloneEngine->CreateCPUStream();
                    communication = standaloneEngine->CreateCPUStream();
                }
                if (settings.method == "gmres")
                    gmres = std::make_unique<GMRES>(prototype, settings.subspace);
                else
                    solution.emplace(Ops::NewVector(prototype));
            }
            template <class F>
            void Stage(F &&run) const
            {
                RunLocalStage(mpi, execution.collectiveChecks,
                              std::forward<F>(run), "linear local phase failed");
            }
            Input &Halo(Field &field)
            {
                const auto found = std::find_if(inputs.begin(), inputs.end(), [&](const auto &input)
                                                { return Ops::SameStorage(input.field, field); });
                DNDS_check_throw_info(found != inputs.end(), "linear operator input has no prepared halo binding");
                return *found;
            }
            void Apply(Field &input, Field &output)
            {
                const LinearProfile::Scope timing(profile, LinearProfile::Action);
                auto engine = compute.LockEngine();
                OperatorSchedule schedule(engine, compute, communication, execution.overlap);
                auto exchange = Exchange(Halo(input), schedule);
                if (execution.overlap)
                    schedule.Submit([&]
                                    { problem->Apply(input, output, LinearRegion::Core); });
                exchange.Await(schedule);
                schedule.Submit([&]
                                { problem->Apply(input, output, execution.overlap ? LinearRegion::HaloDependent : LinearRegion::All); });
                Stage([&]
                      { schedule.Join(); });
            }
            void Sweep(const Field &rhs, Field &input, Field &output, bool knownZero)
            {
                const LinearProfile::Scope timing(profile, LinearProfile::SGS);
                // Iteration scopes outlive the schedule, including exceptional
                // draining of accepted work. No worker enters MPI.
                std::optional<Iteration> iteration;
                auto engine = compute.LockEngine();
                OperatorSchedule schedule(engine, compute, communication, execution.overlap);
                std::optional<Exchange> exchange;
                if (!knownZero)
                    exchange.emplace(Halo(input), schedule);
                schedule.Submit([&]
                                {
                if (knownZero)
                    iteration.emplace(problem->BeginSGSZero(rhs, forward, output, settings.exchangeBetweenSweeps));
                else
                    iteration.emplace(problem->BeginSGS(rhs, input, forward, output, settings.exchangeBetweenSweeps));
                if (knownZero) iteration->Lower();
                else if (execution.overlap) iteration->Lower(LinearRegion::Core); });
                if (!knownZero)
                {
                    exchange->Await(schedule);
                    schedule.Submit([&]
                                    { iteration->Lower(execution.overlap ? LinearRegion::HaloDependent : LinearRegion::All); });
                }
                if (settings.exchangeBetweenSweeps)
                {
                    // Deliberately exposed middle exchange. Portable MPI must
                    // observe the completed forward producer before posting.
                    Stage([&]
                          { schedule.Join(); });
                    Exchange(Halo(forward), schedule).Await(schedule);
                }
                schedule.Submit([&]
                                { iteration->Upper(); iteration->Finish(); });
                Stage([&]
                      { schedule.Join(); });
            }
            void JacobiSweep(const Field &rhs, Field &input, Field &output, bool knownZero)
            {
                auto &first = settings.jacobiLocalSweeps == 2 ? forward : output;
                auto engine = compute.LockEngine();
                OperatorSchedule schedule(engine, compute, communication, execution.overlap);
                if (knownZero)
                    schedule.Submit([&]
                                    { problem->JacobiZero(rhs, first); });
                else
                {
                    auto exchange = Exchange(Halo(input), schedule);
                    if (execution.overlap)
                        schedule.Submit([&]
                                        { problem->Jacobi(rhs, input, first, LinearRegion::Core); });
                    exchange.Await(schedule);
                    schedule.Submit([&]
                                    { problem->Jacobi(rhs, input, first, execution.overlap ? LinearRegion::HaloDependent : LinearRegion::All); });
                }
                if (settings.jacobiLocalSweeps == 2)
                    schedule.Submit([&]
                                    {
                        // Stream ordering completes the first sweep. The second reads
                        // its owned result but the same frozen remote values;
                        // a one-hop core is not proof for a two-hop overlap.
                        std::vector<R::Array<real>> ghosts;
                        Ops::VisitFields(input, [&](const auto &part) { ghosts.push_back(part.Ghost()); });
                        index iField = 0;
                        Ops::VisitFields(forward, [&](auto &part)
                        {
                            if (knownZero)
                            {
                                auto write = part.Ghost().Write();
                                const auto values = write.template View<R::UniformTableRows>();
                                CPU::FillRows(values, values.Width(), 0);
                            }
                            else
                                part.Ghost().CopyFrom(ghosts[iField]).Wait();
                            ++iField;
                        });
                        problem->Jacobi(rhs, forward, output); });
                Stage([&]
                      { schedule.Join(); });
            }
            void ApplyILU(const Field &rhs, Field &output)
            {
                auto engine = compute.LockEngine();
                OperatorSchedule schedule(engine, compute, communication, false);
                // The operation record owns numeric handles and matrix/factors;
                // it never retains this solver (and its execution context).
                schedule.Submit([matrix = problem, source = Ops::Alias(rhs),
                                 scratch = Ops::Alias(forward), target = Ops::Alias(output)]() mutable
                                { matrix->ApplyILU(source, scratch, target); });
                schedule.Join(); // Synchronous preconditioner return boundary.
            }
            void Precondition(const Field &rhs, Field &output)
            {
                if (settings.preconditioner == "ilu")
                {
                    if (settings.iluPreconditionerCorrections == 1)
                    {
                        Stage([&]
                              { ApplyILU(rhs, output); });
                        return;
                    }
                    // Every application begins at zero, preserving a fixed
                    // linear preconditioner for ordinary left GMRES. No outer
                    // nonlinear RHS or changing stopping policy lives here.
                    Stage([&]
                          { ApplyILU(rhs, old); });
                    for (int correction = 1; correction < settings.iluPreconditionerCorrections; ++correction)
                    {
                        Apply(old, action);
                        Stage([&]
                              {
                            Ops::LinearCombination(residual, real(1), rhs, real(-1), action);
                            ApplyILU(residual, action);
                            Ops::LinearCombination(next, real(1), old, real(1), action);
                            Ops::Copy(old, next); });
                    }
                    Stage([&]
                          { Ops::Copy(output, old); });
                }
                else
                {
                    // First sweep consumes no old array. Only later sweeps
                    // populate old from the preceding complete result.
                    for (int sweep = 0; sweep < settings.sgsSweeps; ++sweep)
                    {
                        Sweep(rhs, old, next, sweep == 0);
                        if (sweep + 1 < settings.sgsSweeps)
                            Stage([&]
                                  { Ops::Copy(old, next); });
                    }
                    Stage([&]
                          { Ops::Copy(output, next); });
                }
            }
            static void Clear(Field &field)
            {
                Ops::VisitFields(field, [](auto &part)
                                 {
                for (auto *array : {&part.Owned(), &part.Ghost()})
                {
                    auto write = array->Write();
                    const auto values = write.template View<R::UniformTableRows>();
                    CPU::FillRows(values, values.Width(), 0);
                } });
            }
            std::vector<real> Products(const Field &x, const std::vector<const Field *> &probes) const
            {
                // TODO(sync-free-gmres): replace this host Arnoldi boundary
                // with managed device-aware reduction buffers,
                // then queue the small-matrix/coefficient-update phase behind
                // their completion, instead of exposing this host vector.
                const LinearProfile::Scope timing(profile, LinearProfile::Products);
                std::vector<real> local, global(probes.size());
                Stage([&]
                      { local = Ops::LocalProducts(x, probes, inverseComponentScales); });
                MPI::AllreduceInto<MPI::Reduction::Sum>(
                    MPI::ContiguousView<const real>{local.data(), local.size()},
                    MPI::ContiguousView<real>{global.data(), global.size()}, mpi.comm);
                return global;
            }
            bool Finite(const Field &x) const
            {
                const LinearProfile::Scope timing(profile, LinearProfile::Products);
                int valid = 1;
                Stage([&]
                      { valid = Ops::LocalFinite(x); });
                if (execution.collectiveChecks)
                    valid = MPI::AllreduceValue<MPI::Reduction::Min>(valid, mpi.comm);
                else if (!valid)
                {
                    // Unchecked production execution cannot return a local
                    // failure while peers continue into the next exchange.
                    MPI::Abort(mpi.comm, "nonfinite fixed-count linear correction");
                }
                return valid;
            }
            static void CheckIdle(const Field &state, bool ghosts)
            {
                Ops::VisitFields(state, [&](const auto &field)
                                 { DNDS_check_throw_info(field.Owned().Storage().IsIdle() && (!ghosts || field.Ghost().Storage().IsIdle()),
                                                         "linear input has an active access or pending completion"); });
            }
            const Field &Solution() const { return gmres ? gmres->Solution() : *solution; }
            Linear::LinearIterationReport Solve(
                const Field &rhs, OptionalRef<const Field> initialGuess,
                const std::vector<real> &componentScales)
            {
                struct FinishProfile
                {
                    LinearProfile &profile;
                    int rank;
                    ~FinishProfile() { profile.Record(rank); }
                } finish{profile, mpi.rank};
                const LinearProfile::Scope totalTiming(profile, LinearProfile::Total);
                auto &data = *this;
                std::optional<R::Array<real>::ReadLease> diagonal;
                std::optional<LinearProblemContext::OffDiagonalReadAccess> offDiagonal;
                data.Stage([&]
                           {
            DNDS_check_throw_info(!data.failed, "linear workspace failed; rebuild it before reuse");
            DNDS_check_throw_info(componentScales.empty() || index(componentScales.size()) == problem->Components(),
                                  "linear component scale width differs");
            std::vector<real> inverse;
            inverse.reserve(componentScales.size());
            for (const auto scale : componentScales)
            {
                DNDS_check_throw_info(std::isfinite(scale) && scale > 0 && std::isfinite(1 / scale), "invalid linear component scale");
                inverse.push_back(1 / scale);
            }
            inverseComponentScales = std::move(inverse);
            data.problem->Validate(); rhs.Validate();
            Ops::CheckCompatible(data.old, rhs);
            CheckIdle(rhs, false);
            Ops::VisitFields(rhs, [&](const auto &field)
            {
                DNDS_check_throw_info(!A::Detail::Overlaps(field.Owned().Storage(), data.problem->Diagonal().Storage()) &&
                    (!data.problem->HasAssembledOffDiagonal() || !A::Detail::Overlaps(field.Owned().Storage(), data.problem->OffDiagonal().Values().Storage())),
                    "linear RHS aliases matrix storage");
            });
            if (initialGuess)
            {
                Ops::CheckCompatible(rhs, *initialGuess);
                CheckIdle(*initialGuess, false);
            }
            for (const auto &input : data.inputs)
            {
                input.field.Validate();
                CheckIdle(input.field, true);
                Ops::RequireDistinct(rhs, input.field);
            }
            diagonal.emplace(data.problem->Diagonal().Read());
            offDiagonal.emplace(data.problem->ReadOffDiagonal());
            if (UsesILU(data.settings))
            {
                const LinearProfile::Scope timing(data.profile, LinearProfile::Factor);
                if (!data.problem->ILUFactorsCurrent()) data.problem->FactorizeILU();
            }
            else data.problem->InvertDiagonal(); });
                try
                {
                    if (data.gmres)
                        return data.gmres->Solve(rhs, [&](Field &x, Field &y)
                                                 { data.Apply(x, y); }, [&](const Field &x, Field &y)
                                                 { data.Precondition(x, y); }, [&](const Field &x, const auto &probes)
                                                 { return data.Products(x, probes); }, {data.settings.maxIterations, data.settings.relativeTolerance, data.settings.absoluteTolerance, data.settings.residualCheckInterval}, initialGuess.Get());
                    auto &x = *data.solution;
                    const bool directFirstILU = !initialGuess && data.settings.method == "ilu" && data.settings.stopping == "fixedCount";
                    data.Stage([&]
                               {
                const LinearProfile::Scope timing(data.profile, LinearProfile::Algebra);
                if (initialGuess) Ops::CopyGuess(x, *initialGuess);
                else if (data.settings.method != "sgs" && data.settings.method != "jacobi" && !directFirstILU) Clear(x); });
                    Linear::LinearIterationReport report;
                    // Defect formation is independent of stopping/reporting:
                    // fixed-count ILU needs it only before later corrections.
                    const auto formDefect = [&]
                    {
                        data.Apply(x, data.action);
                        ++report.operatorApplications;
                        data.Stage([&]
                                   { const LinearProfile::Scope timing(data.profile, LinearProfile::Algebra);
                                     Ops::LinearCombination(data.residual, real(1), rhs, real(-1), data.action); });
                    };
                    const auto correct = [&]
                    {
                        if (directFirstILU && report.iterations == 0)
                        {
                            // x0 is analytic zero: neither owned nor ghost
                            // storage is read. Publish M_local^-1*b directly,
                            // avoiding clear, zero+correction and next->x copy.
                            data.Stage([&]
                                       { const LinearProfile::Scope timing(data.profile, LinearProfile::Triangular);
                                         data.ApplyILU(rhs, x); });
                            ++report.preconditionerApplications;
                            ++report.iterations;
                            return;
                        }
                        if (data.settings.method == "ilu")
                        {
                            // Stationary defect correction, with frozen local
                            // factors: x_next = x + M_local^-1 (b - A*x).
                            // A*x establishes its halo; the triangular solve
                            // itself has no communication or Krylov history.
                            const auto &defect = !initialGuess && report.iterations == 0 ? rhs : data.residual;
                            data.Stage([&]
                                       {
                                {
                                    const LinearProfile::Scope timing(data.profile, LinearProfile::Triangular);
                                    data.ApplyILU(defect, data.action);
                                }
                                const LinearProfile::Scope timing(data.profile, LinearProfile::Algebra);
                                Ops::LinearCombination(data.next, real(1), x, real(1), data.action); });
                            ++report.preconditionerApplications;
                        }
                        else if (data.settings.method == "jacobi")
                        {
                            data.JacobiSweep(rhs, x, data.next, !initialGuess && report.iterations == 0);
                            report.jacobiSweeps += data.settings.jacobiLocalSweeps;
                            report.jacobiInputExchanges += bool(initialGuess) || report.iterations != 0;
                        }
                        else
                            data.Sweep(rhs, x, data.next, !initialGuess && report.iterations == 0);
                        data.Stage([&]
                                   { const LinearProfile::Scope timing(data.profile, LinearProfile::Algebra); Ops::Copy(x, data.next); });
                        ++report.iterations;
                    };
                    if (data.settings.stopping == "fixedCount")
                    {
                        while (report.iterations < data.settings.maxIterations)
                        {
                            if (data.settings.method == "ilu" && (initialGuess || report.iterations != 0))
                                formDefect();
                            correct();
                        }
                        if (!initialGuess && report.iterations == 0)
                            data.Stage([&]
                                       { Ops::Fill(x, 0); }); // Return a real zero answer if no sweep ran.
                        report.usable = data.Finite(x);
                        report.stop = report.usable ? Linear::LinearStop::FixedCount : Linear::LinearStop::NonFinite;
                        return report; // Residual diagnostics stay unevaluated (NaN).
                    }
                    const auto norm = [&](const Field &field)
                    { ++report.innerProductBatches; return std::sqrt(data.Products(field, {&field})[0]); };
                    const real normB = norm(rhs), target = std::max(data.settings.absoluteTolerance, data.settings.relativeTolerance * normB);
                    const auto residual = [&]
                    { formDefect(); return norm(data.residual); };
                    report.initialTrueResidual = report.trueResidual = initialGuess ? residual() : normB;
                    while (std::isfinite(report.trueResidual) && report.trueResidual > target && report.iterations < data.settings.maxIterations)
                    {
                        correct();
                        // Dynamic stopping is an explicit host observation
                        // boundary. Between checks the correction sequence is
                        // fixed. ILU still needs b-A*x to compute its next
                        // correction, but need not reduce/observe that norm.
                        if (report.iterations % data.settings.residualCheckInterval == 0 ||
                            report.iterations == data.settings.maxIterations)
                            report.trueResidual = residual();
                        else if (data.settings.method == "ilu")
                            formDefect();
                    }
                    report.converged = std::isfinite(report.trueResidual) && report.trueResidual <= target;
                    if (!initialGuess && report.iterations == 0 && (data.settings.method == "sgs" || data.settings.method == "jacobi"))
                        data.Stage([&]
                                   { Ops::Fill(x, 0); });
                    report.stop = !std::isfinite(report.trueResidual) ? Linear::LinearStop::NonFinite : report.converged ? Linear::LinearStop::Converged
                                                                                                                         : Linear::LinearStop::IterationLimit;
                    report.usable = report.stop != Linear::LinearStop::NonFinite && std::isfinite(norm(x));
                    if (!report.usable)
                    {
                        report.converged = false;
                        report.stop = Linear::LinearStop::NonFinite;
                    }
                    return report;
                }
                catch (...)
                {
                    MPI::AbortIfSolverFailure("Cauchy linear operation failed");
                    data.failed = true;
                    throw;
                }
            }
        };
        template <class Data>
        std::shared_ptr<Data> PrepareSolver(std::shared_ptr<LinearProblemContext> problem, const typename Data::Field &prototype,
                                            const LinearSolver::HaloBinder &bindHalo, LinearSolveSettings settings, LinearExecution execution)
        {
            using Field = typename Data::Field;
            using Ops = typename Data::Ops;
            DNDS_check_throw_info(problem && problem->Space().Distributed(), "linear solver needs a distributed problem context");
            const MPIInfo mpi(problem->Space().Communicator());
            MPI::LocalStage(mpi, [&]
                            {
            problem->Validate(); prototype.Validate();
            DNDS_check_throw_info(prototype.Space().Same(problem->Space()) && Ops::Width(prototype) == problem->Components() && prototype.Device() == R::Device::Host(),
                                  "linear solver prototype differs from its CPU problem");
            DNDS_check_throw_info(bindHalo && (settings.method == "sgs" || settings.method == "jacobi" || settings.method == "ilu" || settings.method == "gmres") &&
                                      (settings.stopping == "residual" || (settings.stopping == "fixedCount" && settings.method != "gmres")) &&
                                      (settings.preconditioner == "sgs" || settings.preconditioner == "ilu") && settings.maxIterations >= 0 && settings.residualCheckInterval > 0 &&
                                      settings.subspace > 0 && settings.subspace < INT_MAX && settings.sgsSweeps > 0 && settings.iluFill >= -1 && settings.iluFill <= 5 &&
                                      settings.iluPreconditionerCorrections > 0 &&
                                      settings.jacobiLocalSweeps >= 1 && settings.jacobiLocalSweeps <= 2 &&
                                      (settings.iluOrdering == "natural" || settings.iluOrdering == "rcm") &&
                                      std::isfinite(settings.relativeTolerance) && settings.relativeTolerance >= 0 &&
                                      std::isfinite(settings.absoluteTolerance) && settings.absoluteTolerance >= 0,
                                  "invalid linear solver controls");
            DNDS_check_throw_info(!((UsesILU(settings) || settings.method == "jacobi") && settings.exchangeBetweenSweeps),
                                  "local ILU/Jacobi has no inter-scan exchange"); });
            const std::array<int64_t, 17> integers{settings.method == "gmres" ? 1 : settings.method == "ilu"  ? 2
                                                                                : settings.method == "jacobi" ? 3
                                                                                                              : 0,
                                                   settings.preconditioner == "ilu", settings.maxIterations, settings.subspace,
                                                   settings.sgsSweeps, settings.iluFill, settings.iluCoreFirst, settings.exchangeBetweenSweeps,
                                                   execution.overlap, execution.collectiveChecks, problem->Components(), int(problem->OffDiagonalMode()), settings.stopping == "fixedCount", settings.iluOrdering == "rcm", settings.iluPreconditionerCorrections, settings.jacobiLocalSweeps, settings.residualCheckInterval};
            auto lowest = integers, highest = integers;
            const std::array<real, 2> tolerances{settings.relativeTolerance, settings.absoluteTolerance};
            auto lowTolerance = tolerances, highTolerance = tolerances;
            MPI::AllreduceInPlace<MPI::Reduction::Min>(lowest, mpi.comm);
            MPI::AllreduceInPlace<MPI::Reduction::Max>(highest, mpi.comm);
            MPI::AllreduceInPlace<MPI::Reduction::Min>(lowTolerance, mpi.comm);
            MPI::AllreduceInPlace<MPI::Reduction::Max>(highTolerance, mpi.comm);
            DNDS_check_throw_info(lowest == highest && lowTolerance == highTolerance, "linear solver controls differ between domain ranks");
            Ops::Agree(prototype, mpi.comm);
            std::shared_ptr<Data> data;
            MPI::LocalStage(mpi, [&]
                            {
            data = std::make_shared<Data>(problem, prototype, settings, execution);
            data->inputs.reserve(size_t(settings.subspace) + 4);
            if (UsesILU(settings)) problem->PrepareILUStructure(settings.iluFill, settings.iluCoreFirst, {}, settings.iluOrdering); });
            const auto bind = [&](Field &field)
            {
                typename Data::Input input{Ops::Alias(field), mpi.comm, {}};
                MPI::LocalStage(mpi, [&]
                                {
                index count = 0;
                Ops::VisitFields(field, [&](auto &) { ++count; });
                input.slots.reserve(size_t(count)); });
                Ops::VisitFields(field, [&](auto &part)
                                 { input.slots.push_back({bindHalo(part), {}}); });
                data->inputs.push_back(std::move(input));
            };
            if (data->gmres)
                data->gmres->VisitOperatorInputs(bind);
            else
                bind(*data->solution);
            if (settings.method == "sgs" || (settings.method == "gmres" && settings.preconditioner == "sgs"))
            {
                bind(data->old);
                bind(data->forward);
            }
            else if (settings.method == "gmres" && settings.preconditioner == "ilu" && settings.iluPreconditionerCorrections > 1)
                bind(data->old);
            return data;
        }
    }
    struct LinearSolver::Data : StateLinearSolverData<FieldLinearOperations>
    {
        Runtime::SessionAccess session;
        using StateLinearSolverData::StateLinearSolverData;
    };
    struct DofLinearSolver::Data : StateLinearSolverData<DofLinearOperations>
    {
        Runtime::SessionAccess session;
        using StateLinearSolverData::StateLinearSolverData;
    };
    LinearSolver::LinearSolver(std::shared_ptr<LinearProblemContext> problem, const Field &prototype, const HaloBinder &bindHalo,
                               LinearSolveSettings settings, LinearExecution execution)
        : _data(PrepareSolver<Data>(std::move(problem), prototype, bindHalo, std::move(settings), execution)) {}
    DofLinearSolver::DofLinearSolver(std::shared_ptr<LinearProblemContext> problem, const EquationDofs &prototype, const HaloBinder &bindHalo,
                                     LinearSolveSettings settings, LinearExecution execution)
        : _data(PrepareSolver<Data>(std::move(problem), prototype, bindHalo, std::move(settings), execution)) {}
    LinearSolver::Field LinearSolver::Solution() const
    {
        DNDS_check_throw_info(_data, "linear solver was moved");
        return _data->Solution();
    }
    EquationDofs DofLinearSolver::Solution() const
    {
        DNDS_check_throw_info(_data, "linear solver was moved");
        return _data->Solution();
    }
    Linear::LinearIterationReport LinearSolver::Solve(
        const Field &rhs, OptionalRef<const Field> initialGuess,
        const std::vector<real> &componentScales)
    {
        DNDS_check_throw_info(_data, "linear solver was moved");
        auto session = _data->session.Acquire();
        return _data->Solve(rhs, initialGuess, componentScales);
    }
    Linear::LinearIterationReport DofLinearSolver::Solve(
        const EquationDofs &rhs, OptionalRef<const EquationDofs> initialGuess,
        const std::vector<real> &componentScales)
    {
        DNDS_check_throw_info(_data, "linear solver was moved");
        auto session = _data->session.Acquire();
        return _data->Solve(rhs, initialGuess, componentScales);
    }
}
