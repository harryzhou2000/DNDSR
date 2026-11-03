#include "Cauchy/Implicit/LinearSolver.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include <array>
#include <optional>

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    namespace A = Algebra;
    namespace
    {
        using Field = LinearSolver::Field;
        Field NewVector(const Field &prototype)
        {
            return Field(prototype.Space(), R::Array<real>(prototype.Owned().Layout()), R::Array<real>(prototype.Ghost().Layout()));
        }
        void Clear(Field &field)
        {
            for (auto *array : {&field.Owned(), &field.Ghost()})
            {
                auto write = array->Write();
                const auto values = write.View<R::UniformTableRows>();
                for (index iRow = 0; iRow < values.RowCount(); ++iRow)
                    for (index j = 0; j < values.RowSize(iRow); ++j)
                        values(iRow, j) = 0;
            }
        }
    }
    struct LinearSolver::Data
    {
        std::shared_ptr<LinearProblemContext> problem;
        MPIInfo mpi;
        LinearSolveSettings settings;
        LinearExecution execution;
        std::unique_ptr<Linear::FieldGMRES> gmres;
        std::optional<Field> solution;
        Field old, next, forward, action, residual;
        struct Input
        {
            Field *field;
            R::TransferBinding halo;
        };
        std::vector<Input> inputs;
        bool failed = false;
        Data(std::shared_ptr<LinearProblemContext> matrix, const Field &prototype, LinearSolveSettings controls, LinearExecution policy)
            : problem(std::move(matrix)), mpi(problem->Space().Communicator()), settings(std::move(controls)), execution(policy),
              old(NewVector(prototype)), next(NewVector(prototype)), forward(NewVector(prototype)), action(NewVector(prototype)), residual(NewVector(prototype))
        {
            if (settings.method == "gmres")
                gmres = std::make_unique<Linear::FieldGMRES>(prototype, settings.subspace);
            else
                solution.emplace(NewVector(prototype));
        }
        template <class F>
        void Stage(F &&run) const
        {
            if (execution.collectiveChecks)
                MPI::LocalStage(mpi, std::forward<F>(run));
            else
                run();
        }
        R::TransferBinding &Halo(Field &field)
        {
            const auto found = std::find_if(inputs.begin(), inputs.end(), [&](const auto &input)
                                            { return input.field == &field; });
            DNDS_check_throw_info(found != inputs.end(), "linear operator input has no prepared halo binding");
            return found->halo;
        }
        void Apply(Field &input, Field &output)
        {
            auto exchange = Halo(input).Begin();
            if (execution.overlap)
                Stage([&]
                      { problem->Apply(input, output, LinearRegion::Core); });
            Stage([&]
                  { exchange.Wait(); problem->Apply(input, output, execution.overlap ? LinearRegion::HaloDependent : LinearRegion::All); });
        }
        void Sweep(const Field &rhs, Field &input, Field &output, bool knownZero)
        {
            std::optional<R::TransferOperation> exchange;
            if (!knownZero)
                exchange.emplace(Halo(input).Begin());
            std::optional<LinearProblemContext::SGSIteration> iteration;
            Stage([&]
                  {
                iteration.emplace(problem->BeginSGS(rhs, input, forward, output, settings.exchangeBetweenSweeps));
                if (knownZero) iteration->Lower();
                else if (execution.overlap) iteration->Lower(LinearRegion::Core); });
            if (!knownZero)
                Stage([&]
                      { exchange->Wait(); iteration->Lower(execution.overlap ? LinearRegion::HaloDependent : LinearRegion::All); });
            if (settings.exchangeBetweenSweeps)
                Stage([&]
                      { Halo(forward).Begin().Wait(); });
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
                              { A::Copy(old, next); });
                }
                Stage([&]
                      { A::Copy(output, next); });
            }
        }
        std::vector<real> Products(const Field &x, const std::vector<const Field *> &probes) const
        {
            std::vector<real> local(probes.size(), 0), global(probes.size());
            // Operand schemas were prepared once and calls are synchronous.
            // No per-dot schema collectives or per-element ownership checks.
            Stage([&]
                  {
                const auto read = x.Owned().Read();
                const auto source = read.View<R::UniformTableRows>();
                std::vector<R::Array<real>::ReadLease> retained;
                std::vector<R::ArrayView<const real, R::AccessBoundsChecks, R::UniformTableRows>> values;
                retained.reserve(probes.size()); values.reserve(probes.size());
                for (const auto *probe : probes)
                {
                    retained.push_back(probe->Owned().Read());
                    values.push_back(retained.back().View<R::UniformTableRows>());
                }
                for (index iRow = 0; iRow < source.RowCount(); ++iRow)
                    for (index j = 0; j < source.RowSize(iRow); ++j)
                        for (index k = 0; k < index(values.size()); ++k) local[k] += source(iRow, j) * values[k](iRow, j); });
            MPI_Allreduce(local.data(), global.data(), int(global.size()), MPI_DOUBLE, MPI_SUM, mpi.comm);
            return global;
        }
    };
    LinearSolver::LinearSolver(std::shared_ptr<LinearProblemContext> problem, const Field &prototype, const HaloBinder &bindHalo,
                               LinearSolveSettings settings, LinearExecution execution)
    {
        DNDS_check_throw_info(problem && problem->Space().Distributed(), "linear solver needs a distributed problem context");
        const MPIInfo mpi(problem->Space().Communicator());
        MPI::LocalStage(mpi, [&]
                        {
            problem->Validate(); prototype.Validate();
            DNDS_check_throw_info(prototype.Space().Same(problem->Space()) && index(prototype.Width()) == problem->Components() && prototype.Device() == R::Device::Host(),
                                  "linear solver prototype differs from its CPU problem");
            DNDS_check_throw_info(bindHalo && (settings.method == "sgs" || settings.method == "gmres") &&
                                      (settings.preconditioner == "sgs" || settings.preconditioner == "ilu") && settings.maxIterations >= 0 &&
                                      settings.subspace > 0 && settings.subspace < INT_MAX && settings.sgsSweeps > 0 && settings.iluFill >= -1 && settings.iluFill <= 5 &&
                                      std::isfinite(settings.relativeTolerance) && settings.relativeTolerance >= 0 &&
                                      std::isfinite(settings.absoluteTolerance) && settings.absoluteTolerance >= 0,
                                  "invalid linear solver controls");
            DNDS_check_throw_info(!(settings.method == "gmres" && settings.preconditioner == "ilu" && settings.exchangeBetweenSweeps),
                                  "local ILU has no inter-scan exchange"); });
        const std::array<int64_t, 12> integers{settings.method == "gmres", settings.preconditioner == "ilu", settings.maxIterations, settings.subspace,
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
        MPI::LocalStage(mpi, [&]
                        {
            _data = std::make_shared<Data>(problem, prototype, settings, execution);
            if (settings.method == "gmres" && settings.preconditioner == "ilu") problem->PrepareILUStructure(settings.iluFill, settings.iluCoreFirst); });
        const auto bind = [&](Field &field)
        { _data->inputs.push_back({&field, bindHalo(field)}); };
        if (_data->gmres)
            _data->gmres->VisitOperatorInputs(bind);
        else
            bind(*_data->solution);
        if (settings.method != "gmres" || settings.preconditioner == "sgs")
        {
            bind(_data->old);
            bind(_data->forward);
        }
    }
    const LinearSolver::Field &LinearSolver::Solution() const
    {
        DNDS_check_throw_info(_data, "linear solver was moved");
        return _data->gmres ? _data->gmres->Solution() : *_data->solution;
    }
    Linear::LinearIterationReport LinearSolver::Solve(const Field &rhs, const Field *initialGuess)
    {
        DNDS_check_throw_info(_data, "linear solver was moved");
        auto &data = *_data;
        std::optional<R::Array<real>::ReadLease> diagonal;
        std::optional<LinearProblemContext::OffDiagonalReadAccess> offDiagonal;
        MPI::LocalStage(data.mpi, [&]
                        {
            DNDS_check_throw_info(!data.failed, "linear workspace failed; rebuild it before reuse");
            data.problem->Validate(); rhs.Validate();
            DNDS_check_throw_info(rhs.Space().Same(data.problem->Space()) && index(rhs.Width()) == data.problem->Components() && rhs.Device() == R::Device::Host(), "linear RHS differs from its problem");
            DNDS_check_throw_info(rhs.Owned().Storage().IsIdle(), "linear RHS must be host-visible and released by other accesses");
            DNDS_check_throw_info(!A::Detail::Overlaps(rhs.Owned().Storage(), data.problem->Diagonal().Storage()) &&
                                      (!data.problem->HasAssembledOffDiagonal() || !A::Detail::Overlaps(rhs.Owned().Storage(), data.problem->OffDiagonal().Values().Storage())), "linear RHS aliases matrix storage");
            if (initialGuess)
            {
                initialGuess->Validate();
                DNDS_check_throw_info(initialGuess->Space().Same(rhs.Space()) && initialGuess->Width() == rhs.Width() && initialGuess->Device() == rhs.Device() &&
                                          initialGuess->Owned().Storage().IsIdle(), "linear initial guess differs or has an active access");
            }
            for (const auto &input : data.inputs)
            {
                DNDS_check_throw_info(input.field->Owned().Storage().IsIdle() && input.field->Ghost().Storage().IsIdle(), "linear workspace input has an active access");
                for (const auto *source : {&rhs.Owned(), &rhs.Ghost()})
                    for (const auto *destination : {&input.field->Owned(), &input.field->Ghost()})
                        DNDS_check_throw_info(!A::Detail::Overlaps(source->Storage(), destination->Storage()), "linear RHS aliases mutable workspace");
            }
            diagonal.emplace(data.problem->Diagonal().Read());
            offDiagonal.emplace(data.problem->ReadOffDiagonal());
            if (data.settings.method == "gmres" && data.settings.preconditioner == "ilu")
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
                else if (x.Owned().Storage().LegacyPointer() != initialGuess->Owned().Storage().LegacyPointer() || !(x.Owned().Layout() == initialGuess->Owned().Layout()))
                    A::Copy(x, *initialGuess); });
            Linear::LinearIterationReport report;
            const auto norm = [&](const Field &field)
            { ++report.innerProductBatches; return std::sqrt(data.Products(field, {&field})[0]); };
            const real normB = norm(rhs), target = std::max(data.settings.absoluteTolerance, data.settings.relativeTolerance * normB);
            const auto residual = [&]
            {
                data.Apply(x, data.action);
                ++report.operatorApplications;
                data.Stage([&]
                           { A::LinearCombination(data.residual, real(1), rhs, real(-1), data.action); });
                return norm(data.residual);
            };
            report.initialTrueResidual = report.trueResidual = initialGuess ? residual() : normB;
            while (std::isfinite(report.trueResidual) && report.trueResidual > target && report.iterations < data.settings.maxIterations)
            {
                data.Sweep(rhs, x, data.next, !initialGuess && report.iterations == 0);
                data.Stage([&]
                           { A::Copy(x, data.next); });
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
}
