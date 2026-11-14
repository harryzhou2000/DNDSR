#include "CFV/Detail/PolynomialSamplingKernels.hpp"
#include "CFV/Plans/PolynomialSamplingPlan.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"

namespace DNDS::CFV
{
    void PolynomialSamplingPlan::EvaluateValues(const FVField &mean, const FVField &coefficients, Runtime::Array<real> &output) const
    {
        if (Device() != Runtime::Device::Host())
        {
            Evaluate(mean, coefficients, output, 0);
            return;
        }
        namespace R = Runtime;
        const index components = mean.Width(), modes = _modes;
        if (_basisRows.RowCount() || _outputRows.RowCount() || _needsGhost || _cachedDerivatives != 1 || (components != 5 && components != 6 && components != 7) ||
            (modes != 2 && modes != 9) || mean.Owned().Layout().Kind() != R::RowLayoutKind::Table ||
            coefficients.Owned().Layout().Kind() != R::RowLayoutKind::Table || output.Layout().Kind() != R::RowLayoutKind::Table)
        {
            Evaluate(mean, coefficients, output);
            return;
        }
        // Same entry checks and lifetime discipline as Evaluate; no ghost
        // lease or source/shape ownership is hidden in the numerical helper.
        auto geometry = _mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        Validate();
        ValidateOutput(mean, coefficients, output, 0);
        auto meanRead = mean.Owned().Read(), recRead = coefficients.Owned().Read();
        auto idsRead = _cells.Read();
        auto basisRead = _basis.Read();
        auto write = output.Write();
        const bool evaluated = Detail::TryEvaluateCachedOwnedTable(idsRead.View<R::FixedRows<1>>(),
                                                                   meanRead.View<R::TableRows>(), recRead.View<R::TableRows>(), basisRead.View<R::TableRows>(), write.View<R::TableRows>());
        DNDS_assert(evaluated);
    }
}
