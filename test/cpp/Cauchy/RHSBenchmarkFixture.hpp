#pragma once

// Private generated workload for the remaining plan53 measurements. No solver
// algorithm lives here: all spatial/source/nonlinear work uses public Cauchy APIs.
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Physics/EquationSetup.hpp"
#include "Cauchy/Physics/EquationProperties.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "CartesianFixture.hpp"

namespace Plan53
{
    namespace C = DNDS::Cauchy;
    namespace R = DNDS::Runtime;
    using I = DNDS::index;
    struct RHSFixture
    {
        static C::DomainSettings Settings(bool mixture, const std::string &limiter)
        {
            C::DomainSettings s;
            s.equation.transport = "NewtonianFourier";
            s.equation.viscosity = .001;
            s.execution.overlap = true;
            s.implicit.collectiveChecks = false;
            s.numerics.reconstruction = "variational";
            s.numerics.reconstructionIterations = 5;
            s.numerics.reconstructionTolerance = 0;
            s.numerics.limiter = limiter;
            s.numerics.positivity = false;
            s.numerics.viscousGradient = "polynomial";
            // The logical-DOF dispatch's supported common CPU profile.
            s.numerics.flux = "rusanov";
            s.numerics.faceFallback = false;
            s.numerics.characteristicLimiter = false;
            s.numerics.viscousLimited = true;
            s.numerics.cfv.maxOrder = 3;
            s.numerics.cfv.intOrder = 5;
            s.numerics.cfv.SORInstead = false;
            if (mixture)
            {
                s.equation.eos = "idealGasMixture";
                s.equation.viscosity = 0;
                s.equation.source = "chemical";
                s.equation.mixture.mechanismFile = "h2o2.yaml";
                s.equation.referenceScales = {{"rho0", 1.225}, {"U0", 340}, {"L0", 2}, {"T0", 288.15}};
            }
            return s;
        }
        static C::EquationDofs State(const C::FiniteVolume &fv, bool mixture)
        {
            if (!mixture)
                return C::EquationDofs::Coupled(fv.Equation().Schema(), fv.Metrics().BuildField(5));
            DNDS::Algebra::DofBundle bundle(fv.Metrics().Space());
            for (const auto &block : fv.Equation().Schema().blocks)
                bundle.Add(block.name, fv.Metrics().BuildField(I(block.components.size())));
            return C::EquationDofs::Separate(fv.Equation().Schema(), std::move(bundle));
        }
        C::DomainSettings settings;
        C::EquationSetup physical;
        C::FiniteVolume fv;
        C::EquationDofs state, residual;
        std::vector<uint64_t> versions;
        RHSFixture(const DNDS::MPIInfo &mpi, I width, bool mixture, const std::string &limiter)
            : settings(Settings(mixture, limiter)), physical(C::EquationSetup::Resolve(settings.equation)),
              fv(CauchyTest::PeriodicQuads(mpi, width), settings, physical.equation),
              state(State(fv, mixture)), residual(C::EquationDofOperations::NewVector(state))
        {
            {
                auto write = state.WriteOwned();
                const auto flow = write.Block("flow");
                if (!mixture)
                    for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                    {
                        const auto p = fv.Metrics().GetCellCent(iCell);
                        const auto u = physical.equation.Gas().Conservative({1 + .01 * std::sin(6.28 * p[0]), .2, -.1, .05, 1});
                        for (int j = 0; j < 5; ++j)
                            flow(iCell, j) = u[j];
                    }
                else
                {
                    DNDS_check_throw_info(physical.workers && physical.equation.Mixture().SpeciesNumber() == 10 && state.PhysicalFieldNumber() == 2,
                                          "assessment requires the explicit ten-species h2o2 backend and split fields");
                    const auto rhoY = write.Block("species");
                    auto evaluator = physical.equation.MakePointEvaluator(physical.workers->Acquire(), {});
                    const auto fractions = physical.equation.IndependentMassFractions({{"H2", .027}, {"O2", .22}, {"H2O", .02}});
                    std::vector<double> species(fractions.size());
                    for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                    {
                        const auto p = fv.Metrics().GetCellCent(iCell);
                        const double rho = .8 + .001 * std::sin(6.28 * p[0]);
                        for (I k = 0; k < I(species.size()); ++k)
                            species[k] = rho * fractions[k];
                        const auto u = evaluator.FromPrimitive({rho, .2, -.04, .03, 0}, {species.data(), int(species.size())},
                                                               1400. / physical.equation.Mixture().Scales().T0);
                        for (int j = 0; j < 5; ++j)
                            flow(iCell, j) = u[j];
                        for (I k = 0; k < I(species.size()); ++k)
                            rhoY(iCell, k) = species[k];
                    }
                }
            }
            for (I iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                versions.push_back(state.PhysicalField(iField).Owned().Version());
        }
        void CheckFrozen() const
        {
            for (I iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                DNDS_check_throw_info(state.PhysicalField(iField).Owned().Version() == versions[iField], "assessment changed frozen means");
        }
        static std::vector<double> Values(const C::EquationDofs &value)
        {
            std::vector<double> result;
            auto read = value.ReadOwned();
            for (I iBlock = 0; iBlock < value.BlockNumber(); ++iBlock)
            {
                const auto rows = read.Block(iBlock);
                for (I iCell = 0; iCell < I(value.Space().OwnedRows()); ++iCell)
                    for (I j = 0; j < rows.ComponentNumber(); ++j)
                    {
                        DNDS_check_throw_info(std::isfinite(rows(iCell, j)), "nonfinite assessment field");
                        result.push_back(rows(iCell, j));
                    }
            }
            return result;
        }
    };
}
