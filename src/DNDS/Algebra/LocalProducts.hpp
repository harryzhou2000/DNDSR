#pragma once

#include "DNDS/Core/Defines.hpp"
#include <vector>

namespace DNDS::Algebra::Detail
{
    /// Private CPU batch-product arithmetic over already acquired views.
    /// Ownership/schema validation stays in each field/equation adapter. Keep
    /// row -> logical component -> probe accumulation order for reproducibility;
    /// no layout flattening, communication, allocation or per-element dispatch.
    template <class Source, class Probe>
    void AccumulateLocalProducts(Source source, const std::vector<Probe> &probes,
                                 index rows, index components, std::vector<real> &products)
    {
        for (index iRow = 0; iRow < rows; ++iRow)
            for (index component = 0; component < components; ++component)
                for (index iProbe = 0; iProbe < index(probes.size()); ++iProbe)
                    products[iProbe] += source(iRow, component) * probes[iProbe](iRow, component);
    }
}
