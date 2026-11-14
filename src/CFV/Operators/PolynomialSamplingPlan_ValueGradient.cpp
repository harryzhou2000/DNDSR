#include "DNDS/MPI/Failure.hpp"
#include "CFV/Plans/PolynomialSamplingPlan.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "CFV/Detail/PolynomialSamplingKernels.hpp"

namespace DNDS::CFV
{
    namespace R = Runtime;

    void PolynomialSamplingPlan::EvaluateValueGradient(const FVField &mean, const FVField &coefficients,
                                                       const PolynomialSampleOutputs &outputs) const
    {
        if (Device() != Runtime::Device::Host())
        {
            EvaluateDeviceValueGradient(mean, coefficients, outputs);
            return;
        }
        auto geometry = _mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        Validate();
        const index components = mean.Width(), modes = _modes;
        const int dimension = _dimension;
        ValidateField(mean, components);
        ValidateField(coefficients, components * modes);
        const std::array<R::Array<real> *, 4> arrays{
            outputs.values, outputs.gradients[0], outputs.gradients[1], outputs.gradients[2]};
        // Access is tracked by payload buffer, independently of metadata
        // handles. Byte overlap alone misses aliases of zero-byte storage.
        const auto overlaps = [](const R::Array<real> &a, const R::Array<real> &b)
        {
            return a.Identity() == b.Identity() || Algebra::Detail::Overlaps(a.Storage(), b.Storage());
        };
        bool any = false;
        for (int iOutput = 0; iOutput < 4; ++iOutput)
        {
            if (!arrays[iOutput])
                continue;
            any = true;
            const auto &output = *arrays[iOutput];
            DNDS_check_throw_info(iOutput == 0 || (iOutput <= _maximumDerivative && iOutput <= dimension),
                                  "polynomial gradient output is outside the permitted directions");
            DNDS_check_throw_info(output.GetDevice() == R::Device::Host() && !output.Layout().IsVariable() &&
                                      output.RowCount() == Size() && output.Layout().Width() == components,
                                  "polynomial sample bundle output shape/placement mismatch");
            for (const auto *input : {&mean.Owned(), &mean.Ghost(), &coefficients.Owned(), &coefficients.Ghost()})
                DNDS_check_throw_info(!overlaps(output, *input),
                                      "polynomial sample output overlaps an input");
            for (int iPrevious = 0; iPrevious < iOutput; ++iPrevious)
                DNDS_check_throw_info(!arrays[iPrevious] || !overlaps(output, *arrays[iPrevious]),
                                      "polynomial sample outputs overlap");
        }
        DNDS_check_throw_info(any, "polynomial sampling needs at least one output");
        // Deliberately bounded specialization. The generic evaluator remains
        // the reference for other layouts, cache policies, widths and ghosts.
        bool table = mean.Owned().Layout().Kind() == R::RowLayoutKind::Table &&
                     coefficients.Owned().Layout().Kind() == R::RowLayoutKind::Table;
        for (const auto *output : arrays)
            table &= !output || output->Layout().Kind() == R::RowLayoutKind::Table;
        if (_basisRows.RowCount() || _outputRows.RowCount() || _needsGhost || dimension != 2 || components != 6 || modes != 9 || _cachedDerivatives != 3 ||
            !arrays[1] || !arrays[2] || !table)
        {
            for (int iOutput = 0; iOutput < 4; ++iOutput)
                if (arrays[iOutput])
                    Evaluate(mean, coefficients, *arrays[iOutput], iOutput);
            return;
        }

        const auto meanRead = mean.Owned().Read(), coefficientRead = coefficients.Owned().Read();
        const auto cellRead = _cells.Read();
        const auto basisRead = _basis.Read();
        std::array<std::optional<R::Array<real>::WriteLease>, 3> writes;
        try
        {
            for (int iOutput = 0; iOutput < 3; ++iOutput)
                if (arrays[iOutput])
                    writes[iOutput].emplace(arrays[iOutput]->Write());
        }
        catch (...)
        {
            MPI::AbortIfSolverFailure("CFV source sampling failed");
            for (auto &write : writes)
                if (write)
                    write->CancelUnmodified();
            throw;
        }
        try
        {
            const auto coefficientsView = coefficientRead.View<R::FixedRows<54>>();
            const auto cells = cellRead.View<R::FixedRows<1>>();
            const auto basis = basisRead.View<R::FixedRows<27>>();
            const auto dx = writes[1]->View<R::FixedRows<6>>();
            const auto dy = writes[2]->View<R::FixedRows<6>>();
            const auto means = meanRead.View<R::FixedRows<6>>();
            const auto values = arrays[0] ? writes[0]->View<R::FixedRows<6>>() : Detail::SamplingFixedOutput<6>{};
            Detail::EvaluateCachedOwnedValueGradient(cells, means, coefficientsView, basis, values, dx, dy, bool(arrays[0]));
        }
        catch (...)
        {
            MPI::AbortIfSolverFailure("CFV source sampling failed");
            for (auto &write : writes)
                if (write)
                    write->Fail();
            throw;
        }
    }
}
