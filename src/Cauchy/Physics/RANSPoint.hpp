#pragma once

#include "Physics/RANS/Model.hpp"

namespace DNDS::Cauchy
{
    /// Gather into the shared bounded point scratch. Logical flow/RANS row
    /// views preserve AoS/SoA/AoSoA strides. Derivative accessors are indexed
    /// (component,direction), as in Cauchy; shared RANS stores the transpose.
    /// Caller supplies molecular properties, scalar wall distance and cell
    /// length explicitly. This performs no EOS evaluation, clipping or leasing.
    template <class Flow, class Scalars, class FlowDerivative, class ScalarDerivative>
    Physics::RANS::Point PrepareRANSPoint(int components, const Flow &flow, const Scalars &scalars,
                                          const FlowDerivative &flowDerivative, const ScalarDerivative &scalarDerivative,
                                          real molecularViscosity, real cp, real gamma, real wallDistance, real cellLength)
    {
        Physics::RANS::Point point;
        for (int j = 0; j < 5; ++j)
        {
            point.state[j] = flow[j];
            for (int d = 0; d < 3; ++d)
                point.gradient(d, j) = flowDerivative(j, d);
        }
        for (int j = 0; j < components; ++j)
        {
            point.state[5 + j] = scalars[j];
            for (int d = 0; d < 3; ++d)
                point.gradient(d, 5 + j) = scalarDerivative(j, d);
        }
        point.molecularViscosity = molecularViscosity;
        point.cp = cp;
        point.gamma = gamma;
        point.wallDistance = wallDistance;
        point.cellLength = cellLength;
        return point;
    }
}
