#pragma once

#include "DNDS/Algebra/DofAlgebra.hpp"
#include <Eigen/QR>
#include <functional>
#include <limits>

namespace DNDS::Linear
{
    enum class LinearStop
    {
        Converged,
        IterationLimit,
        ArnoldiBreakdown,
        PreconditionerBreakdown,
        NonFinite
    };
    struct KrylovSettings
    {
        index maxIterations = 40;
        real relativeTolerance = 1e-8, absoluteTolerance = 1e-12;
    };
    struct LinearIterationReport
    {
        LinearStop stop = LinearStop::IterationLimit;
        bool converged = false, usable = false;
        index iterations = 0, restarts = 0, operatorApplications = 0, preconditionerApplications = 0, innerProductBatches = 0;
        /// NaN means that diagnostic was not evaluated before the stop.
        real initialTrueResidual = std::numeric_limits<real>::quiet_NaN();
        real trueResidual = std::numeric_limits<real>::quiet_NaN(), preconditionedResidual = std::numeric_limits<real>::quiet_NaN();
    };

    /// Restarted left-GMRES on native DOF fields. Operator/preconditioner and
    /// global inner products are supplied by the caller; no mesh, halo or
    /// physical-time policy is hidden in the mathematical iteration. Callbacks
    /// complete synchronously and coordinate distributed failures themselves.
    /// CPU profile initially; unavailable device profiles reject explicitly.
    class FieldGMRES
    {
    public:
        using Field = Algebra::DofField<real>;
        using Operator = std::function<void(Field &, Field &)>;
        using Preconditioner = std::function<void(const Field &, Field &)>;
        /// Batched products <x,probe[k]>. Supply one fixed positive-definite
        /// inner product throughout a solve. The caller owns its communicator.
        using Products = std::function<std::vector<real>(const Field &, const std::vector<const Field *> &)>;

    private:
        int _subspace;
        Field _solution, _action, _residual, _preconditioned, _work;
        std::vector<Field> _basis;
        MatrixXR _h;
        VectorXR _g, _cosine, _sine;
        static int Subspace(int count)
        {
            DNDS_check_throw_info(count > 0 && count < INT_MAX, "invalid GMRES subspace size");
            return count;
        }
        static Field NewVector(const Field &prototype)
        {
            prototype.Validate();
            DNDS_check_throw_info(prototype.Device() == Runtime::Device::Host(), "FieldGMRES CPU profile cannot execute device vectors");
            return Field(prototype.Space(), Runtime::Array<real>(prototype.Owned().Layout()), Runtime::Array<real>(prototype.Ghost().Layout()));
        }
        void Check(const Field &field) const
        {
            field.Validate();
            DNDS_check_throw_info(field.Space().Same(_solution.Space()) && field.Width() == _solution.Width() && field.Device() == _solution.Device(),
                                  "GMRES field space, width or backend differs");
        }

    public:
        FieldGMRES(const Field &prototype, int subspace)
            : _subspace(Subspace(subspace)), _solution(NewVector(prototype)), _action(NewVector(prototype)),
              _residual(NewVector(prototype)), _preconditioned(NewVector(prototype)), _work(NewVector(prototype)),
              _h(subspace + 1, subspace), _g(subspace + 1), _cosine(subspace), _sine(subspace)
        {
            _basis.reserve(size_t(subspace) + 1);
            for (int i = 0; i <= subspace; ++i)
                _basis.push_back(NewVector(prototype));
        }
        FieldGMRES(const FieldGMRES &) = delete;
        FieldGMRES &operator=(const FieldGMRES &) = delete;
        /// Preparation hook: bind persistent halos for exactly these mutable
        /// operator inputs. Their object addresses/storage stay fixed. Do not
        /// change layouts, retain conflicting views or mutate them during Solve.
        void VisitOperatorInputs(const std::function<void(Field &)> &visit)
        {
            visit(_solution);
            for (auto &field : _basis)
                visit(field);
        }
        /// Borrowed result; the next Solve overwrites it. Inspect the returned
        /// report: convergence requires a true-residual check, and usable also
        /// permits a finite iteration-limited inexact solve. Other stop reasons
        /// must not be applied as an accepted nonlinear correction.
        const Field &Solution() const { return _solution; }
        LinearIterationReport Solve(const Field &rhs, const Operator &apply, const Preconditioner &precondition,
                                    const Products &products, KrylovSettings settings = {}, const Field *initialGuess = nullptr)
        {
            Check(rhs);
            if (initialGuess)
                Check(*initialGuess);
            DNDS_check_throw_info(apply && precondition && products && settings.maxIterations >= 0 &&
                                      std::isfinite(settings.relativeTolerance) && settings.relativeTolerance >= 0 &&
                                      std::isfinite(settings.absoluteTolerance) && settings.absoluteTolerance >= 0,
                                  "invalid GMRES callbacks or controls");
            LinearIterationReport report;
            const auto finish = [&](LinearStop stop)
            {
                if ((stop == LinearStop::Converged || stop == LinearStop::IterationLimit) && (initialGuess || report.iterations > 0))
                {
                    // A callback can annihilate a bad direction. A small true
                    // residual alone must not label a nonfinite iterate usable.
                    ++report.innerProductBatches;
                    const auto value = products(_solution, {&_solution});
                    DNDS_check_throw_info(value.size() == 1, "GMRES solution-norm batch size differs");
                    if (!(value[0] >= 0) || !std::isfinite(value[0]))
                        stop = LinearStop::NonFinite;
                }
                report.stop = stop;
                report.converged = stop == LinearStop::Converged;
                report.usable = report.converged || stop == LinearStop::IterationLimit;
                return report;
            };
            const auto norm = [&](const Field &field)
            {
                ++report.innerProductBatches;
                const auto value = products(field, {&field});
                DNDS_check_throw_info(value.size() == 1, "GMRES inner-product batch size differs");
                return value[0] >= 0 ? std::sqrt(value[0]) : std::numeric_limits<real>::quiet_NaN();
            };
            const auto action = [&](Field &input, Field &output)
            { apply(input, output); ++report.operatorApplications; };
            const auto pc = [&](const Field &input, Field &output)
            { precondition(input, output); ++report.preconditionerApplications; };
            if (!initialGuess)
                Algebra::Fill(_solution, real(0));
            else if (initialGuess != &_solution)
                Algebra::Copy(_solution, *initialGuess);
            const real normB = norm(rhs);
            if (!std::isfinite(normB))
                return finish(LinearStop::NonFinite);
            const real target = std::max(settings.absoluteTolerance, settings.relativeTolerance * normB);
            if (initialGuess)
            {
                action(_solution, _action);
                Algebra::LinearCombination(_residual, real(1), rhs, real(-1), _action);
            }
            else
                Algebra::Copy(_residual, rhs); // A is linear and x is exactly zero.
            report.initialTrueResidual = report.trueResidual = norm(_residual);
            if (!std::isfinite(report.trueResidual))
                return finish(LinearStop::NonFinite);
            if (report.trueResidual <= target)
                return finish(LinearStop::Converged);
            if (!settings.maxIterations)
                return finish(LinearStop::IterationLimit);
            pc(_residual, _preconditioned);
            report.preconditionedResidual = norm(_preconditioned);
            real preconditionedB = report.preconditionedResidual;
            if (initialGuess)
            {
                pc(rhs, _work);
                preconditionedB = norm(_work);
            }
            const real leftTarget = std::max(settings.relativeTolerance * preconditionedB,
                                             normB > 0 ? settings.absoluteTolerance * (preconditionedB / normB) : real(0));
            if (!std::isfinite(preconditionedB) || !std::isfinite(leftTarget))
                return finish(LinearStop::NonFinite);
            std::vector<const Field *> probes;
            probes.reserve(size_t(_subspace) + 1);
            while (report.iterations < settings.maxIterations)
            {
                const real beta = report.preconditionedResidual;
                if (!std::isfinite(beta))
                    return finish(LinearStop::NonFinite);
                if (beta == 0)
                    return finish(LinearStop::PreconditionerBreakdown);
                Algebra::Copy(_basis[0], _preconditioned);
                Algebra::Scale(_basis[0], 1 / beta);
                _h.setZero();
                _g.setZero();
                _g[0] = beta;
                int used = 0;
                bool closed = false;
                for (int j = 0; j < _subspace && report.iterations < settings.maxIterations; ++j)
                {
                    action(_basis[j], _action);
                    pc(_action, _work);
                    probes.clear();
                    for (int i = 0; i <= j; ++i)
                        probes.push_back(&_basis[i]);
                    real beforeNorm = 0;
                    for (int pass = 0; pass < 2; ++pass)
                    {
                        if (!pass)
                            probes.push_back(&_work);
                        ++report.innerProductBatches;
                        auto projection = products(_work, probes);
                        DNDS_check_throw_info(projection.size() == probes.size(), "GMRES projection batch size differs");
                        for (auto value : projection)
                            if (!std::isfinite(value))
                                return finish(LinearStop::NonFinite);
                        if (!pass)
                        {
                            if (projection.back() < 0)
                                return finish(LinearStop::NonFinite);
                            beforeNorm = std::sqrt(projection.back());
                            probes.pop_back();
                        }
                        for (int i = 0; i <= j; ++i)
                        {
                            _h(i, j) += projection[i];
                            Algebra::Axpy(_work, -projection[i], _basis[i]);
                        }
                    }
                    const real next = norm(_work);
                    if (!std::isfinite(next))
                        return finish(LinearStop::NonFinite);
                    _h(j + 1, j) = next;
                    closed = next <= 32 * std::numeric_limits<real>::epsilon() * beforeNorm;
                    if (!closed)
                    {
                        Algebra::Copy(_basis[j + 1], _work);
                        Algebra::Scale(_basis[j + 1], 1 / next);
                    }
                    for (int i = 0; i < j; ++i)
                    {
                        const real first = _h(i, j), second = _h(i + 1, j);
                        _h(i, j) = _cosine[i] * first + _sine[i] * second;
                        _h(i + 1, j) = -_sine[i] * first + _cosine[i] * second;
                    }
                    const real radius = std::hypot(_h(j, j), _h(j + 1, j));
                    _cosine[j] = radius > 0 ? _h(j, j) / radius : 1;
                    _sine[j] = radius > 0 ? _h(j + 1, j) / radius : 0;
                    _h(j, j) = radius;
                    _h(j + 1, j) = 0;
                    _g[j + 1] = -_sine[j] * _g[j];
                    _g[j] *= _cosine[j];
                    ++report.iterations;
                    used = j + 1;
                    if (closed || std::abs(_g[j + 1]) <= leftTarget)
                        break;
                }
                const auto leastSquares = _h.topLeftCorner(used + 1, used).colPivHouseholderQr();
                // A zero operator produces a rank-zero projected system. It
                // is an algebraic breakdown, not a nonfinite update direction.
                if (!leastSquares.rank())
                    return finish(LinearStop::ArnoldiBreakdown);
                const VectorXR coefficients = leastSquares.solve(_g.head(used + 1));
                if (!coefficients.allFinite())
                    return finish(LinearStop::NonFinite);
                for (int i = 0; i < used; ++i)
                    Algebra::Axpy(_solution, coefficients[i], _basis[i]);
                action(_solution, _action);
                Algebra::LinearCombination(_residual, real(1), rhs, real(-1), _action);
                report.trueResidual = norm(_residual);
                if (!std::isfinite(report.trueResidual))
                    return finish(LinearStop::NonFinite);
                pc(_residual, _preconditioned);
                report.preconditionedResidual = norm(_preconditioned);
                if (!std::isfinite(report.preconditionedResidual))
                    return finish(LinearStop::NonFinite);
                if (report.trueResidual <= target)
                    return finish(LinearStop::Converged);
                if (closed)
                    return finish(LinearStop::ArnoldiBreakdown);
                if (report.iterations < settings.maxIterations)
                    ++report.restarts;
            }
            return finish(LinearStop::IterationLimit);
        }
    };
}
