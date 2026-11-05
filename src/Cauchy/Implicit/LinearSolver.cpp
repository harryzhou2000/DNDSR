#include "Cauchy/Implicit/LinearSolver.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Cauchy/State/EquationDofCommunication.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include <array>
#include <optional>
#include <utility>

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    namespace A = Algebra;
    namespace
    {
        bool UsesILU(const LinearSolveSettings &settings)
        {
            return settings.method == "ilu" || (settings.method == "gmres" && settings.preconditioner == "ilu");
        }
        struct FieldLinearOperations : Linear::FieldKrylovOperations
        {
            using State = Field;
            using Iteration = LinearProblemContext::SGSIteration;
            static index Width(const State &state) { return index(state.Width()); }
            static void Agree(const State &, MPI_Comm) {}
            template <class S, class F>
            static void VisitFields(S &state, F &&visit) { visit(state); }
            static std::vector<real> LocalProducts(const State &x, const std::vector<const State *> &probes)
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
                for (index iRow = 0; iRow < source.RowCount(); ++iRow)
                    for (index j = 0; j < source.RowSize(iRow); ++j)
                        for (index k = 0; k < index(values.size()); ++k)
                            local[k] += source(iRow, j) * values[k](iRow, j);
                return local;
            }
        };
        struct EquationLinearOperations : EquationDofOperations
        {
            using Iteration = LinearProblemContext::EquationSGSIteration;
            static index Width(const State &state) { return state.ComponentNumber(); }
            static void Agree(const State &state, MPI_Comm communicator) { AgreeEquationBinding(state, communicator); }
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
                        Retire();
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
                if (execution.collectiveChecks)
                    MPI::LocalStage(mpi, std::forward<F>(run));
                else
                    run();
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
                std::optional<Exchange> exchange;
                if (!knownZero)
                    exchange.emplace(Exchange(Halo(input)));
                std::optional<Iteration> iteration;
                Stage([&]
                      {
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
            void Precondition(const Field &rhs, Field &output)
            {
                if (settings.preconditioner == "ilu")
                    Stage([&]
                          { problem->ApplyILU(rhs, forward, output); });
                else
                {
                    Stage([&]
                          { Clear(old); });
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
                std::vector<real> local, global(probes.size());
                Stage([&]
                      { local = Ops::LocalProducts(x, probes); });
                MPI_Allreduce(local.data(), global.data(), int(global.size()), MPI_DOUBLE, MPI_SUM, mpi.comm);
                return global;
            }
            static void CheckIdle(const Field &state, bool ghosts)
            {
                Ops::VisitFields(state, [&](const auto &field)
                                 { DNDS_check_throw_info(field.Owned().Storage().IsIdle() && (!ghosts || field.Ghost().Storage().IsIdle()),
                                                         "linear input has an active access or pending completion"); });
            }
            const Field &Solution() const { return gmres ? gmres->Solution() : *solution; }
            Linear::LinearIterationReport Solve(const Field &rhs, const Field *initialGuess)
            {
                auto &data = *this;
                std::optional<R::Array<real>::ReadLease> diagonal;
                std::optional<LinearProblemContext::OffDiagonalReadAccess> offDiagonal;
                MPI::LocalStage(data.mpi, [&]
                                {
            DNDS_check_throw_info(!data.failed, "linear workspace failed; rebuild it before reuse");
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
                input.field->Validate();
                CheckIdle(*input.field, true);
                Ops::RequireDistinct(rhs, *input.field);
            }
            diagonal.emplace(data.problem->Diagonal().Read());
            offDiagonal.emplace(data.problem->ReadOffDiagonal());
            if (UsesILU(data.settings))
            {
                if (!data.problem->ILUFactorsCurrent()) data.problem->FactorizeILU();
            }
            else data.problem->InvertDiagonal(); });
                try
                {
                    if (data.gmres)
                        return data.gmres->Solve(rhs, [&](Field &x, Field &y)
                                                 { data.Apply(x, y); }, [&](const Field &x, Field &y)
                                                 { data.Precondition(x, y); }, [&](const Field &x, const auto &probes)
                                                 { return data.Products(x, probes); }, {data.settings.maxIterations, data.settings.relativeTolerance, data.settings.absoluteTolerance}, initialGuess);
                    auto &x = *data.solution;
                    data.Stage([&]
                               {
                if (!initialGuess) Clear(x);
                else Ops::CopyGuess(x, *initialGuess); });
                    Linear::LinearIterationReport report;
                    const auto norm = [&](const Field &field)
                    { ++report.innerProductBatches; return std::sqrt(data.Products(field, {&field})[0]); };
                    const real normB = norm(rhs), target = std::max(data.settings.absoluteTolerance, data.settings.relativeTolerance * normB);
                    const auto residual = [&]
                    {
                        data.Apply(x, data.action);
                        ++report.operatorApplications;
                        data.Stage([&]
                                   { Ops::LinearCombination(data.residual, real(1), rhs, real(-1), data.action); });
                        return norm(data.residual);
                    };
                    report.initialTrueResidual = report.trueResidual = initialGuess ? residual() : normB;
                    while (std::isfinite(report.trueResidual) && report.trueResidual > target && report.iterations < data.settings.maxIterations)
                    {
                        if (data.settings.method == "ilu")
                        {
                            // Stationary defect correction, with frozen local
                            // factors: x_next = x + M_local^-1 (b - A*x).
                            // A*x establishes its halo; the triangular solve
                            // itself has no communication or Krylov history.
                            const auto &defect = !initialGuess && report.iterations == 0 ? rhs : data.residual;
                            data.Stage([&]
                                       {
                                data.problem->ApplyILU(defect, data.forward, data.action);
                                Ops::LinearCombination(data.next, real(1), x, real(1), data.action); });
                            ++report.preconditionerApplications;
                        }
                        else
                            data.Sweep(rhs, x, data.next, !initialGuess && report.iterations == 0);
                        data.Stage([&]
                                   { Ops::Copy(x, data.next); });
                        ++report.iterations;
                        report.trueResidual = residual();
                    }
                    report.converged = std::isfinite(report.trueResidual) && report.trueResidual <= target;
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
            DNDS_check_throw_info(bindHalo && (settings.method == "sgs" || settings.method == "ilu" || settings.method == "gmres") &&
                                      (settings.preconditioner == "sgs" || settings.preconditioner == "ilu") && settings.maxIterations >= 0 &&
                                      settings.subspace > 0 && settings.subspace < INT_MAX && settings.sgsSweeps > 0 && settings.iluFill >= -1 && settings.iluFill <= 5 &&
                                      std::isfinite(settings.relativeTolerance) && settings.relativeTolerance >= 0 &&
                                      std::isfinite(settings.absoluteTolerance) && settings.absoluteTolerance >= 0,
                                  "invalid linear solver controls");
            DNDS_check_throw_info(!(UsesILU(settings) && settings.exchangeBetweenSweeps),
                                  "local ILU has no inter-scan exchange"); });
            const std::array<int64_t, 12> integers{settings.method == "gmres" ? 1 : settings.method == "ilu" ? 2
                                                                                                             : 0,
                                                   settings.preconditioner == "ilu", settings.maxIterations, settings.subspace,
                                                   settings.sgsSweeps, settings.iluFill, settings.iluCoreFirst, settings.exchangeBetweenSweeps,
                                                   execution.overlap, execution.collectiveChecks, problem->Components(), int(problem->OffDiagonalMode())};
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
            if (UsesILU(settings)) problem->PrepareILUStructure(settings.iluFill, settings.iluCoreFirst); });
            const auto bind = [&](Field &field)
            {
                typename Data::Input input{&field, {}};
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
            return data;
        }
    }
    struct LinearSolver::Data : StateLinearSolverData<FieldLinearOperations>
    {
        using StateLinearSolverData::StateLinearSolverData;
    };
    struct EquationLinearSolver::Data : StateLinearSolverData<EquationLinearOperations>
    {
        using StateLinearSolverData::StateLinearSolverData;
    };
    LinearSolver::LinearSolver(std::shared_ptr<LinearProblemContext> problem, const Field &prototype, const HaloBinder &bindHalo,
                               LinearSolveSettings settings, LinearExecution execution)
        : _data(PrepareSolver<Data>(std::move(problem), prototype, bindHalo, std::move(settings), execution)) {}
    EquationLinearSolver::EquationLinearSolver(std::shared_ptr<LinearProblemContext> problem, const EquationDofs &prototype, const HaloBinder &bindHalo,
                                               LinearSolveSettings settings, LinearExecution execution)
        : _data(PrepareSolver<Data>(std::move(problem), prototype, bindHalo, std::move(settings), execution)) {}
    const LinearSolver::Field &LinearSolver::Solution() const
    {
        DNDS_check_throw_info(_data, "linear solver was moved");
        return _data->Solution();
    }
    const EquationDofs &EquationLinearSolver::Solution() const
    {
        DNDS_check_throw_info(_data, "linear solver was moved");
        return _data->Solution();
    }
    Linear::LinearIterationReport LinearSolver::Solve(const Field &rhs, const Field *initialGuess)
    {
        DNDS_check_throw_info(_data, "linear solver was moved");
        return _data->Solve(rhs, initialGuess);
    }
    Linear::LinearIterationReport EquationLinearSolver::Solve(const EquationDofs &rhs, const EquationDofs *initialGuess)
    {
        DNDS_check_throw_info(_data, "linear solver was moved");
        return _data->Solve(rhs, initialGuess);
    }
}
