#include "Cauchy/Implicit/LinearSolver.hpp"
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
        template <class Rows>
        bool FiniteRows(Rows values, index components)
        {
            bool valid = true;
            for (index iRow = 0; iRow < values.RowCount(); ++iRow)
                for (index j = 0; j < components; ++j)
                    valid &= std::isfinite(values(iRow, j));
            return valid;
        }
        struct FieldLinearOperations : Linear::FieldKrylovOperations
        {
            using State = Field;
            using Iteration = LinearProblemContext::SGSIteration;
            static index Width(const State &state) { return index(state.Width()); }
            static void Agree(const State &, MPI_Comm) {}
            static bool LocalFinite(const State &state)
            {
                const auto read = state.Owned().Read();
                return FiniteRows(read.View<R::UniformTableRows>(), Width(state));
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
            static void Agree(const State &state, MPI_Comm communicator) { AgreeEquationBinding(state, communicator); }
            static bool LocalFinite(const State &state)
            {
                const auto read = state.ReadOwned();
                const auto layout = read.ComponentLayout();
                return FiniteRows(layout.View(), Width(state));
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
            std::unique_ptr<GMRES> gmres;
            std::optional<Field> solution;
            Field old, next, forward, action, residual;
            struct Input
            {
                Field *field;
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
            /// Slots are allocated at binding time. Normal retirement drains;
            /// active transfer retirement during unwinding fails immediately.
            /// No per-application allocation or pending request survives this
            /// synchronous controller operation.
            class Exchange
            {
                Input *_input = nullptr;
                void Retire() noexcept
                {
                    if (!_input)
                        return;
                    for (auto &slot : _input->slots)
                        slot.operation.reset();
                    _input->active = false;
                    _input = nullptr;
                }

            public:
                explicit Exchange(Input &input) : _input(&input)
                {
                    DNDS_check_throw_info(!input.active, "linear halo is already active");
                    input.active = true;
                    try
                    {
                        for (auto &slot : input.slots)
                            slot.operation.emplace(slot.binding.Begin());
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
                Exchange(Exchange &&other) noexcept : _input(std::exchange(other._input, nullptr)) {}
                ~Exchange() { Retire(); }
                void Wait()
                {
                    if (!_input)
                        return;
                    for (auto &slot : _input->slots)
                        slot.operation->Wait();
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
                                                { return input.field == &field; });
                DNDS_check_throw_info(found != inputs.end(), "linear operator input has no prepared halo binding");
                return *found;
            }
            void Apply(Field &input, Field &output)
            {
                const LinearProfile::Scope timing(profile, LinearProfile::Action);
                auto exchange = Exchange(Halo(input));
                if (execution.overlap)
                    Stage([&]
                          { problem->Apply(input, output, LinearRegion::Core); });
                exchange.Wait();
                Stage([&]
                      { problem->Apply(input, output, execution.overlap ? LinearRegion::HaloDependent : LinearRegion::All); });
            }
            void Sweep(const Field &rhs, Field &input, Field &output, bool knownZero)
            {
                const LinearProfile::Scope timing(profile, LinearProfile::SGS);
                std::optional<Exchange> exchange;
                if (!knownZero)
                    exchange.emplace(Exchange(Halo(input)));
                std::optional<Iteration> iteration;
                Stage([&]
                      {
                if (knownZero)
                    iteration.emplace(problem->BeginSGSZero(rhs, forward, output, settings.exchangeBetweenSweeps));
                else
                    iteration.emplace(problem->BeginSGS(rhs, input, forward, output, settings.exchangeBetweenSweeps));
                if (knownZero) iteration->Lower();
                else if (execution.overlap) iteration->Lower(LinearRegion::Core); });
                if (!knownZero)
                {
                    exchange->Wait();
                    Stage([&]
                          { iteration->Lower(execution.overlap ? LinearRegion::HaloDependent : LinearRegion::All); });
                }
                if (settings.exchangeBetweenSweeps)
                    Exchange(Halo(forward)).Wait();
                Stage([&]
                      { iteration->Upper(); iteration->Finish(); });
            }
            void JacobiSweep(const Field &rhs, Field &input, Field &output, bool knownZero)
            {
                auto &first = settings.jacobiLocalSweeps == 2 ? forward : output;
                if (knownZero)
                    Stage([&]
                          { problem->JacobiZero(rhs, first); });
                else
                {
                    auto exchange = Exchange(Halo(input));
                    if (execution.overlap)
                        Stage([&]
                              { problem->Jacobi(rhs, input, first, LinearRegion::Core); });
                    exchange.Wait();
                    Stage([&]
                          { problem->Jacobi(rhs, input, first, execution.overlap ? LinearRegion::HaloDependent : LinearRegion::All); });
                }
                if (settings.jacobiLocalSweeps == 2)
                    Stage([&]
                          {
                        // The first sweep is fully joined. The second reads
                        // its owned result but the same frozen remote values;
                        // a one-hop core is not proof for a two-hop overlap.
                        std::vector<const R::Array<real> *> ghosts;
                        Ops::VisitFields(input, [&](const auto &part) { ghosts.push_back(&part.Ghost()); });
                        index iField = 0;
                        Ops::VisitFields(forward, [&](auto &part)
                        {
                            if (knownZero)
                            {
                                auto write = part.Ghost().Write();
                                const auto values = write.template View<R::UniformTableRows>();
                                for (index iRow = 0; iRow < values.RowCount(); ++iRow)
                                    for (index j = 0; j < values.RowSize(iRow); ++j)
                                        values(iRow, j) = 0;
                            }
                            else
                                part.Ghost().CopyFrom(*ghosts[iField]).Wait();
                            ++iField;
                        });
                        problem->Jacobi(rhs, forward, output); });
            }
            void Precondition(const Field &rhs, Field &output)
            {
                if (settings.preconditioner == "ilu")
                {
                    if (settings.iluPreconditionerCorrections == 1)
                    {
                        Stage([&]
                              { problem->ApplyILU(rhs, forward, output); });
                        return;
                    }
                    // Every application begins at zero, preserving a fixed
                    // linear preconditioner for ordinary left GMRES. No outer
                    // nonlinear RHS or changing stopping policy lives here.
                    Stage([&]
                          { problem->ApplyILU(rhs, forward, old); });
                    for (int correction = 1; correction < settings.iluPreconditionerCorrections; ++correction)
                    {
                        Apply(old, action);
                        Stage([&]
                              {
                            Ops::LinearCombination(residual, real(1), rhs, real(-1), action);
                            problem->ApplyILU(residual, forward, action);
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
                    for (index iRow = 0; iRow < values.RowCount(); ++iRow)
                        for (index j = 0; j < values.RowSize(iRow); ++j) values(iRow, j) = 0;
                } });
            }
            std::vector<real> Products(const Field &x, const std::vector<const Field *> &probes) const
            {
                const LinearProfile::Scope timing(profile, LinearProfile::Products);
                std::vector<real> local, global(probes.size());
                Stage([&]
                      { local = Ops::LocalProducts(x, probes, inverseComponentScales); });
                MPI_Allreduce(local.data(), global.data(), int(global.size()), MPI_DOUBLE, MPI_SUM, mpi.comm);
                return global;
            }
            bool Finite(const Field &x) const
            {
                const LinearProfile::Scope timing(profile, LinearProfile::Products);
                int valid = 1;
                Stage([&]
                      { valid = Ops::LocalFinite(x); });
                if (execution.collectiveChecks)
                    MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, mpi.comm);
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
                const Field &rhs, std::optional<std::reference_wrapper<const Field>> initialGuess,
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
                Ops::CheckCompatible(rhs, initialGuess->get());
                CheckIdle(initialGuess->get(), false);
            }
            for (const auto &input : data.inputs)
            {
                input.field->Validate();
                CheckIdle(*input.field, true);
                Ops::RequireDistinct(rhs, *input.field);
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
                                                 { return data.Products(x, probes); }, {data.settings.maxIterations, data.settings.relativeTolerance, data.settings.absoluteTolerance}, initialGuess ? &initialGuess->get() : nullptr);
                    auto &x = *data.solution;
                    const bool directFirstILU = !initialGuess && data.settings.method == "ilu" && data.settings.stopping == "fixedCount";
                    data.Stage([&]
                               {
                const LinearProfile::Scope timing(data.profile, LinearProfile::Algebra);
                if (initialGuess) Ops::CopyGuess(x, initialGuess->get());
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
                                         data.problem->ApplyILU(rhs, data.forward, x); });
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
                                    data.problem->ApplyILU(defect, data.forward, data.action);
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
                        report.trueResidual = residual();
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
                                      (settings.preconditioner == "sgs" || settings.preconditioner == "ilu") && settings.maxIterations >= 0 &&
                                      settings.subspace > 0 && settings.subspace < INT_MAX && settings.sgsSweeps > 0 && settings.iluFill >= -1 && settings.iluFill <= 5 &&
                                      settings.iluPreconditionerCorrections > 0 &&
                                      settings.jacobiLocalSweeps >= 1 && settings.jacobiLocalSweeps <= 2 &&
                                      (settings.iluOrdering == "natural" || settings.iluOrdering == "rcm") &&
                                      std::isfinite(settings.relativeTolerance) && settings.relativeTolerance >= 0 &&
                                      std::isfinite(settings.absoluteTolerance) && settings.absoluteTolerance >= 0,
                                  "invalid linear solver controls");
            DNDS_check_throw_info(!((UsesILU(settings) || settings.method == "jacobi") && settings.exchangeBetweenSweeps),
                                  "local ILU/Jacobi has no inter-scan exchange"); });
            const std::array<int64_t, 16> integers{settings.method == "gmres" ? 1 : settings.method == "ilu"  ? 2
                                                                                : settings.method == "jacobi" ? 3
                                                                                                              : 0,
                                                   settings.preconditioner == "ilu", settings.maxIterations, settings.subspace,
                                                   settings.sgsSweeps, settings.iluFill, settings.iluCoreFirst, settings.exchangeBetweenSweeps,
                                                   execution.overlap, execution.collectiveChecks, problem->Components(), int(problem->OffDiagonalMode()), settings.stopping == "fixedCount", settings.iluOrdering == "rcm", settings.iluPreconditionerCorrections, settings.jacobiLocalSweeps};
            auto lowest = integers, highest = integers;
            const std::array<real, 2> tolerances{settings.relativeTolerance, settings.absoluteTolerance};
            auto lowTolerance = tolerances, highTolerance = tolerances;
            MPI_Allreduce(MPI_IN_PLACE, lowest.data(), int(lowest.size()), MPI_INT64_T, MPI_MIN, mpi.comm);
            MPI_Allreduce(MPI_IN_PLACE, highest.data(), int(highest.size()), MPI_INT64_T, MPI_MAX, mpi.comm);
            MPI_Allreduce(MPI_IN_PLACE, lowTolerance.data(), 2, MPI_DOUBLE, MPI_MIN, mpi.comm);
            MPI_Allreduce(MPI_IN_PLACE, highTolerance.data(), 2, MPI_DOUBLE, MPI_MAX, mpi.comm);
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
                typename Data::Input input{&field, mpi.comm, {}};
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
        const Field &rhs, std::optional<std::reference_wrapper<const Field>> initialGuess,
        const std::vector<real> &componentScales)
    {
        DNDS_check_throw_info(_data, "linear solver was moved");
        auto session = _data->session.Acquire();
        return _data->Solve(rhs, initialGuess, componentScales);
    }
    Linear::LinearIterationReport DofLinearSolver::Solve(
        const EquationDofs &rhs, std::optional<std::reference_wrapper<const EquationDofs>> initialGuess,
        const std::vector<real> &componentScales)
    {
        DNDS_check_throw_info(_data, "linear solver was moved");
        auto session = _data->session.Acquire();
        return _data->Solve(rhs, initialGuess, componentScales);
    }
}
