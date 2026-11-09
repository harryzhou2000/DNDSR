#include "Cauchy/Implicit/EquationJacobianKernels.hpp"

namespace DNDS::Cauchy::Detail
{
    namespace
    {
        template <int Column>
        void AddColumn(FrozenFluxRow6 mean, FrozenFluxRow6 pressureDerivative, real pressure,
                       const Vector3 &normal, const FlowState &speedDerivative,
                       real normalVelocity, real relativeVelocity, real scalarSpeed,
                       real factor, real lambda, FrozenFluxMatrix6 output)
        {
            real deltaSpeed = 0;
            if constexpr (Column < 5)
                deltaSpeed = speedDerivative[Column];
            const real dp = pressureDerivative[Column];
            output[6 * Column] += factor * (mean[0] * deltaSpeed + (Column == 0 ? relativeVelocity - lambda : 0));
            for (int d = 0; d < 3; ++d)
                output[6 * Column + 1 + d] += factor * (mean[1 + d] * deltaSpeed + dp * normal[d] +
                                                        (Column == 1 + d ? relativeVelocity - lambda : 0));
            output[6 * Column + 4] += factor * ((mean[4] + pressure) * deltaSpeed + dp * normalVelocity +
                                                (Column == 4 ? relativeVelocity - lambda : 0));
            output[6 * Column + 5] += factor * (mean[5] * deltaSpeed + (Column == 5 ? scalarSpeed - lambda : 0));
        }
    }

    void AddFrozenFluxBlock6(FrozenFluxRow6 mean, FrozenFluxRow6 pressureDerivative, real pressure,
                             const Vector3 &normal, const Vector3 &gridVelocity,
                             real factor, real lambda, FrozenFluxMatrix6 output)
    {
        const real inverseDensity = 1 / mean[0];
        real normalVelocity = 0, gridNormalVelocity = 0, scalarSpeed = 0;
        for (int d = 0; d < 3; ++d)
        {
            const real velocity = mean[1 + d] / mean[0];
            normalVelocity += velocity * normal[d];
            gridNormalVelocity += gridVelocity[d] * normal[d];
            scalarSpeed += (velocity - gridVelocity[d]) * normal[d];
        }
        const real relativeVelocity = normalVelocity - gridNormalVelocity;
        FlowState speedDerivative{};
        speedDerivative[0] = -normalVelocity * inverseDensity;
        for (int d = 0; d < 3; ++d)
            speedDerivative[1 + d] = normal[d] * inverseDensity;

            // Bounded, operator-owned specialization: no shape/layout product or
            // duplicate mesh/physics preparation template is instantiated.
#define DNDS_ADD_FLUX_COLUMN(column) AddColumn<column>(mean, pressureDerivative, pressure, normal, speedDerivative, normalVelocity, relativeVelocity, scalarSpeed, factor, lambda, output)
        DNDS_ADD_FLUX_COLUMN(0);
        DNDS_ADD_FLUX_COLUMN(1);
        DNDS_ADD_FLUX_COLUMN(2);
        DNDS_ADD_FLUX_COLUMN(3);
        DNDS_ADD_FLUX_COLUMN(4);
        DNDS_ADD_FLUX_COLUMN(5);
#undef DNDS_ADD_FLUX_COLUMN
    }
}
