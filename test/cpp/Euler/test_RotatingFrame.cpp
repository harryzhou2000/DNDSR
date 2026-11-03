#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "Euler/SourceTermContributor.hpp"

namespace
{
    template <DNDS::Euler::EulerModel Model>
    void CheckAbsoluteRotation()
    {
        using namespace DNDS::Euler;
        using Traits = EulerModelTraits<Model>;
        constexpr int dim = Traits::dim;
        constexpr int variables = dim + 5; // Flow plus three untouched scalars.
        typename Traits::TU state(variables);
        state.setConstant(0.1);
        state[0] = 1.7;
        state[1] = 1.2;
        state[2] = -0.8;
        if constexpr (dim == 3)
            state[3] = 0.4;
        state[dim + 1] = 7.5;
        typename Traits::TDiffU gradient(dim, variables);
        gradient.setZero();
        RotatingFrameContributor<Model> frame;
        frame.enabled = true;
        frame.rpm = 120;
        frame.center = {0.3, -0.2, 0.1};
        if constexpr (Traits::gDim == 3)
            frame.axis = DNDS::Geom::tPoint{1, 2, 3}.normalized();
        const auto omega = frame.vOmega();
        const DNDS::Geom::tPoint point{3, -4, 2};
        typename Traits::TU expected(variables);
        expected.setZero();
        // Independent component formula for -Omega cross momentum.
        const DNDS::real mx = state[1], my = state[2];
        const DNDS::real mz = dim == 3 ? state[3] : 0;
        expected[1] = omega[2] * my - omega[1] * mz;
        expected[2] = omega[0] * mz - omega[2] * mx;
        if constexpr (dim == 3)
            expected[3] = omega[1] * mx - omega[0] * my;
        typename Traits::TJacobianU expectedJac(variables, variables);
        expectedJac.setZero();
        expectedJac(1, 2) = -omega[2]; // Stored matrix is -d(source)/dU.
        expectedJac(2, 1) = omega[2];
        if constexpr (dim == 3)
        {
            expectedJac(1, 3) = omega[1];
            expectedJac(2, 3) = -omega[0];
            expectedJac(3, 1) = -omega[1];
            expectedJac(3, 2) = omega[0];
        }
        for (int mode : {0, 1, 2})
        {
            INFO("dimension=" << dim << ", source mode=" << mode);
            typename Traits::TU output(variables);
            typename Traits::TJacobianU jacobian(variables, variables);
            output.setConstant(0.125);
            jacobian.setConstant(0.25);
            frame.evaluate(output, jacobian, state, gradient, point, SourceCellAux{}, 0, 0, mode);
            const auto sourceChange = (output.array() - 0.125).matrix().eval();
            const auto matrixChange = (jacobian.array() - 0.25).matrix().eval();
            if (mode == 0 || mode == 2)
                CHECK(sourceChange.isApprox(expected, 2e-14));
            else
                CHECK(sourceChange.isZero());
            if (mode == 2)
                CHECK(matrixChange.isApprox(expectedJac, 2e-14));
            else
                CHECK(matrixChange.isZero());
        }
        // Finite differences validate all columns, including density, energy
        // and trailing scalar columns that must not gain rotation derivatives.
        constexpr DNDS::real epsilon = 1e-6;
        for (int j = 0; j < variables; ++j)
        {
            typename Traits::TU plus = state, minus = state;
            plus[j] += epsilon;
            minus[j] -= epsilon;
            typename Traits::TU sourcePlus(variables), sourceMinus(variables);
            typename Traits::TJacobianU unused(variables, variables);
            sourcePlus.setZero();
            sourceMinus.setZero();
            unused.setZero();
            frame.evaluate(sourcePlus, unused, plus, gradient, point, SourceCellAux{}, 0, 0, 0);
            frame.evaluate(sourceMinus, unused, minus, gradient, point, SourceCellAux{}, 0, 0, 0);
            CHECK(((sourcePlus - sourceMinus) / (2 * epsilon) + expectedJac.col(j)).norm() < 1e-7);
        }
        for (bool enabled : {false, true})
        {
            frame.enabled = enabled;
            frame.rpm = enabled ? 0 : 120;
            typename Traits::TU output(variables);
            typename Traits::TJacobianU jacobian(variables, variables);
            output.setZero();
            jacobian.setZero();
            frame.evaluate(output, jacobian, state, gradient, point, SourceCellAux{}, 0, 0, 2);
            CHECK(output.isZero());
            CHECK(jacobian.isZero());
        }
    }
}

TEST_CASE("Extended Euler rotation uses absolute momentum and its negative-source Jacobian")
{
    SUBCASE("NS_EX") { CheckAbsoluteRotation<DNDS::Euler::NS_EX>(); }
    SUBCASE("NS_EX_3D") { CheckAbsoluteRotation<DNDS::Euler::NS_EX_3D>(); }
}
