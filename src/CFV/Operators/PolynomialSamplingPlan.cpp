#include "CFV/Plans/PolynomialSamplingPlan.hpp"
#include "CFV/Detail/PolynomialSamplingKernels.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"

namespace DNDS::CFV
{
    namespace R = Runtime;
    void PolynomialSamplingPlan::ValidateOutput(const FVField &mean, const FVField &coefficients, const R::Array<real> &output, int derivative) const
    {
        ValidateField(mean, mean.Width());
        ValidateField(coefficients, mean.Width() * _modes);
        DNDS_check_throw_info(derivative >= 0 && derivative <= (_cacheDerivatives ? _modes : 0) && output.GetDevice() == Device() &&
                                  !output.Layout().IsVariable() && output.RowCount() == Size() && output.Layout().Width() == index(mean.Width()),
                              "polynomial sample output shape/placement or derivative mismatch");
        for (const auto *input : {&mean.Owned(), &mean.Ghost(), &coefficients.Owned(), &coefficients.Ghost()})
            DNDS_check_throw_info(output.Identity() != input->Identity() && !Algebra::Detail::Overlaps(output.Storage(), input->Storage()),
                                  "polynomial sample output overlaps an input");
        if (Device().space == R::MemorySpace::CUDA)
        {
            for (const auto *array : {&mean.Owned(), &mean.Ghost(), &coefficients.Owned(), &coefficients.Ghost(), &output})
                DNDS_check_throw_info(array->Layout().Kind() == R::RowLayoutKind::Table, "CUDA sampling currently requires AoS tables");
        }
    }
    void PolynomialSamplingPlan::Evaluate(const FVField &mean, const FVField &coefficients, R::Array<real> &output, int derivative) const
    {
        ValidateOutput(mean, coefficients, output, derivative);
        if (Device() != R::Device::Host())
        {
#ifndef DNDS_USE_CUDA
            DNDS_check_throw_info(false, "CUDA polynomial sampling is not built");
#else
            R::EagerEngine engine;
            const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(Device().ordinal));
            Evaluate(engine, stream, mean, coefficients, output, derivative);
            engine.Close(); // Explicitly synchronous convenience; queued overload does not join.
#endif
            return;
        }
        const auto geometryGuard = _mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        Validate();
        const auto meanRead = mean.Owned().Read(), coefficientRead = coefficients.Owned().Read();
        const auto meanGhost = (_needsGhost ? mean.Ghost() : _empty).Read();
        const auto coefficientGhost = (_needsGhost ? coefficients.Ghost() : _empty).Read();
        const auto cells = _cells.Read();
        const auto points = _points.Read(), basis = _basis.Read(), moments = _moments.Read(), geometry = _cellGeometry.Read();
        auto writer = output.Write();
        Detail::PolynomialSamplingView plan{cells.View<R::FixedRows<1>>(), points.View<R::FixedRows<3>>(),
                                            basis.View<R::TableRows>(), moments.View<R::TableRows>(), geometry.View<R::FixedRows<8>>(),
                                            Math::TaylorBasis(_dimension, _degree).KernelParameters(), _cachedDerivatives};
        Detail::EvaluatePolynomialSamples(plan,
                                          {meanRead.View<R::UniformTableRows>(), meanGhost.View<R::UniformTableRows>()},
                                          {coefficientRead.View<R::UniformTableRows>(), coefficientGhost.View<R::UniformTableRows>()},
                                          writer.View<R::UniformTableRows>(), derivative);
    }
    void PolynomialSamplingPlan::Evaluate(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                          const FVField &mean, const FVField &coefficients, R::Array<real> &output, int derivative) const
    {
        ValidateOutput(mean, coefficients, output, derivative);
        DNDS_check_throw_info(stream.Context().device == Device(), "sampling stream and plan placement differ");
#ifndef DNDS_USE_CUDA
        DNDS_check_throw_info(Device() == R::Device::Host(), "CUDA polynomial sampling is not built");
#endif
        auto geometryGuard = std::make_shared<Geom::MeshReadAccess>(_mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux()));
        Validate();
        // SubmitArrays' retirement record retains the captured geometry guard,
        // while declared operands retain all sampling payloads. Repeated value
        // refreshes therefore need no engine-wide retained-owner accumulation.
        const auto polynomial = Math::TaylorBasis(_dimension, _degree).KernelParameters();
        R::SubmitArrays(engine, stream, [geometryGuard, polynomial, derivative, cachedDerivatives = _cachedDerivatives](R::ExecutionContext context, auto cells, auto points, auto basis, auto moments, auto geometry, auto meanOwned, auto meanGhost, auto coefficientOwned, auto coefficientGhost, auto result)
                        {
                            Detail::PolynomialSamplingView plan{cells, points, basis, moments, geometry, polynomial, cachedDerivatives};
                            const Detail::SamplingField mean{meanOwned, meanGhost}, coefficients{coefficientOwned, coefficientGhost};
                            if (context.device == R::Device::Host())
                                Detail::EvaluatePolynomialSamples(plan, mean, coefficients, result, derivative);
#ifdef DNDS_USE_CUDA
                            else
                                Detail::LaunchPolynomialSamples(context, plan, mean, coefficients, result, derivative);
#endif
                        },
                        R::ReadAs<R::FixedRows<1>>(_cells), R::ReadAs<R::FixedRows<3>>(_points), R::ReadAs<R::TableRows>(_basis), R::ReadAs<R::TableRows>(_moments), R::ReadAs<R::FixedRows<8>>(_cellGeometry), R::ReadAs<R::UniformTableRows>(mean.Owned()), R::ReadAs<R::UniformTableRows>(_needsGhost ? mean.Ghost() : _empty), R::ReadAs<R::UniformTableRows>(coefficients.Owned()), R::ReadAs<R::UniformTableRows>(_needsGhost ? coefficients.Ghost() : _empty), R::WriteAs<R::UniformTableRows>(output));
    }
    void PolynomialSamplingPlan::EvaluateDeviceValueGradient(const FVField &mean, const FVField &coefficients, const PolynomialSampleOutputs &outputs) const
    {
        const std::array<R::Array<real> *, 4> arrays{outputs.values, outputs.gradients[0], outputs.gradients[1], outputs.gradients[2]};
        bool any = false;
        for (int i = 0; i < 4; ++i)
            if (arrays[i])
            {
                any = true;
                DNDS_check_throw_info(i == 0 || i <= _dimension, "polynomial gradient direction is outside geometry dimension");
                ValidateOutput(mean, coefficients, *arrays[i], i);
                for (int j = 0; j < i; ++j)
                    DNDS_check_throw_info(!arrays[j] || (arrays[j]->Identity() != arrays[i]->Identity() &&
                                                         !Algebra::Detail::Overlaps(arrays[j]->Storage(), arrays[i]->Storage())),
                                          "polynomial sample outputs overlap");
            }
        DNDS_check_throw_info(any, "polynomial sampling needs at least one output");
#ifndef DNDS_USE_CUDA
        DNDS_check_throw_info(false, "CUDA polynomial sampling is not built");
#else
        R::EagerEngine engine;
        const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(Device().ordinal));
        for (int i = 0; i < 4; ++i)
            if (arrays[i])
                Evaluate(engine, stream, mean, coefficients, *arrays[i], i);
        engine.Close();
#endif
    }
}
