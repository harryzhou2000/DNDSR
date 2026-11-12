#include "Cauchy/Physics/EquationSystem.hpp"
#include "Cauchy/Physics/PointProperties.hpp"

namespace DNDS::Cauchy
{
    PointEvaluator EquationSystem::MakePointEvaluator(Physics::Chemistry::ChemicalWorkerPool::Lease worker,
                                                      Physics::Thermodynamics::MixturePropertyRequest request) const
    {
        return PointEvaluator(*this, std::move(worker), request);
    }
}
