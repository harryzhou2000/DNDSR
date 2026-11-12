#include "Cauchy/Bindings/OperatorsBindInternal.hpp"
#include "Cauchy/Core/Domain.hpp"

namespace DNDS::Cauchy::Bindings
{
    FiniteVolumeOperators::FiniteVolumeOperators(const Domain &domain)
        : fv(domain.Discretization()), workers(domain.ChemicalWorkers())
    {
    }
}
