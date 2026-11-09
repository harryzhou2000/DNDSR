#pragma once

#include "DNDS/Core/Defines.hpp"
#include <vector>

namespace DNDS::Algebra::Detail
{
    /// Same logical component scaling for combined and separately stored DOFs.
    /// Caller validates scale width; the empty case performs no scale loads.
    template <class Source, class Probe>
    real AccumulateLocalProduct(Source source, Probe probe, index rows, index components,
                                const std::vector<real> &inverseScales)
    {
        real product = 0;
        for (index iRow = 0; iRow < rows; ++iRow)
            for (index component = 0; component < components; ++component)
                product += inverseScales.empty() ? source(iRow, component) * probe(iRow, component)
                                                 : (source(iRow, component) * inverseScales[component]) *
                                                       (probe(iRow, component) * inverseScales[component]);
        return product;
    }
    /// Private CPU batch-product arithmetic over already acquired views.
    /// Ownership/schema validation stays in each field/equation adapter. Keep
    /// row -> logical component -> probe accumulation order for reproducibility;
    /// no layout flattening, communication, allocation or per-element dispatch.
    template <class Source, class Probe>
    void AccumulateLocalProducts(Source source, const std::vector<Probe> &probes,
                                 index rows, index components, std::vector<real> &products,
                                 const std::vector<real> &inverseScales = {})
    {
        for (index iRow = 0; iRow < rows; ++iRow)
            for (index component = 0; component < components; ++component)
                for (index iProbe = 0; iProbe < index(probes.size()); ++iProbe)
                    products[iProbe] += inverseScales.empty() ? source(iRow, component) * probes[iProbe](iRow, component)
                                                              : (source(iRow, component) * inverseScales[component]) *
                                                                    (probes[iProbe](iRow, component) * inverseScales[component]);
    }
}
