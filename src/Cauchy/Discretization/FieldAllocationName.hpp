#pragma once

#include "DNDS/Algebra/DofField.hpp"

namespace DNDS::Cauchy::FiniteVolumeDetail
{
    inline void NameField(const Algebra::DofField<real> &field, const std::string &name, Runtime::AllocationSite site)
    {
        field.Owned().SetName(name + "/owned", site);
        field.Ghost().SetName(name + "/ghost", site);
    }
}
