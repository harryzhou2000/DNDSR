#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Cauchy/Physics/StateInput.hpp"
#include "Cauchy/Physics/EquationSetup.hpp"
#include "Cauchy/Physics/EquationRANS.hpp"
#include "Euler/Physics/PhysicsProperties.hpp"
#include <cstdlib>

namespace C = DNDS::Cauchy;
namespace P = DNDS::Physics;
namespace E = DNDS::Euler;

TEST_CASE("Tagged physical state conversions preserve Euler variants and distinct RANS scales")
{
    const P::ReferenceScales scales{1.7, 230, 3.2, 310};
    for (const auto *model : {"none", "SA", "KOmegaWilcox", "KOmegaSST", "RealizableKE"})
    {
        C::EquationSettings settings;
        settings.transport = "NewtonianFourier";
        settings.viscosity = .001;
        settings.gasConstant = 287. / scales.HeatCapacity();
        settings.rans.model = model;
        const auto equation = C::EquationSystem::Resolve(settings);
        E::EulerEvaluatorSettings<E::NS_EX_3D>::IdealGasProperty gas;
        gas.rho0 = scales.rho0;
        gas.U0 = scales.U0;
        gas.L0 = scales.L0;
        gas.T0 = scales.T0;
        gas.gamma = settings.gamma;
        gas.Rgas = 287;
        E::PhysicsProperties<E::NS_EX_3D> legacy(gas);
        const auto rm = std::string(model) == "none" ? E::RANS_None : std::string(model) == "SA"         ? E::RANS_SA
                                                                  : std::string(model) == "KOmegaWilcox" ? E::RANS_KOWilcox
                                                                  : std::string(model) == "KOmegaSST"    ? E::RANS_KOSST
                                                                                                         : E::RANS_RKE;
        legacy.setRANS(rm);
        const int nr = equation.HasRANS() ? equation.RANS().ComponentNumber() : 0;
        E::PhysicsProperties<E::NS_EX_3D>::TU primitive(5 + nr), expected;
        primitive.head<5>() << .43, .12, -.03, .02, .8;
        if (nr)
            primitive[5] = .017;
        if (nr == 2)
            primitive[6] = .23;
        legacy.primToConservative(primitive, expected);
        for (const auto *type : {"cons", "consSensible", "primRhoP", "primRhoT", "primTP", "cons_phy", "consSensible_phy", "primRhoP_phy", "primRhoT_phy", "primTP_phy"})
        {
            CAPTURE(model);
            CAPTURE(type);
            E::PhysicsProperties<E::NS_EX_3D>::TU input;
            legacy.conservativeToStateValueOrigin(expected, input, E::StateValueOriginFromName(type));
            const C::Json tagged{{"type", type}, {"state", std::vector<double>(input.data(), input.data() + input.size())}};
            const auto actual = C::ResolvePhysicalState(tagged, equation, scales);
            for (DNDS::index j = 0; j < expected.size(); ++j)
                CHECK(actual.conservative[j] == doctest::Approx(expected[j]).epsilon(2e-12));
            CHECK(actual.primitive[0] == doctest::Approx(.43));
            if (std::string(type) == "consSensible")
                CHECK(C::ResolvePhysicalState(tagged["state"], equation, scales).conservative == actual.conservative);
        }
    }
}

TEST_CASE("Initial boundary and reference adapters share conversion without density fallback")
{
    C::DomainSettings settings;
    settings.equation.transport = "NewtonianFourier";
    settings.equation.viscosity = .001;
    settings.equation.rans.model = "KOmegaSST";
    settings.numerics.faceFallback = false;
    settings.numerics.viscousGradient = "polynomial";
    settings.equation.referenceScales = {{"rho0", 2}, {"U0", 3}, {"L0", 4}, {"T0", 5}};
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    settings.initialState = settings.referenceState = C::Json{{"type", "primRhoP_phy"}, {"state", {.6, .9, 0, 0, 18, .18, .3}}};
    settings.referenceState["state"][0] = .9; // Deliberately distinct from initial rho and rho0.
    settings.boundaries["101"].kind = "farfield";
    settings.boundaries["101"].state = settings.initialState;
    const auto normalized = C::ResolvePhysicalStateInputs(settings, equation, {});
    CHECK(normalized.initialConservative[0] == doctest::Approx(.3));
    CHECK(normalized.referenceConservative[0] == doctest::Approx(.45));
    CHECK(normalized.settings.initialPrimitive == normalized.settings.boundaries.at("101").primitive);
    CHECK(normalized.settings.initialRANS == normalized.settings.boundaries.at("101").ransConservative);
    CHECK(normalized.settings.numerics.ransState.referenceConservative.at("rho") == doctest::Approx(.45));
    const auto policy = C::ResolveRANSReconstructionPolicy(equation, normalized.settings.numerics.ransState, 1e-12);
    CHECK(policy.compressionFloors[0] / DNDS::smallReal == doctest::Approx(.02).epsilon(1e-13));
    CHECK(policy.compressionFloors[1] / DNDS::smallReal == doctest::Approx(.4).epsilon(1e-13));
    const auto parsed = C::SimulationConfig::Resolve({{"defaults", C::Json(settings)}}, 1);
    CHECK(parsed.Domains()[0].settings.referenceState == settings.referenceState);
    auto bad = settings;
    bad.referenceState["state"][0] = 0;
    CHECK_THROWS(C::ResolvePhysicalStateInputs(bad, equation, {}));
    bad = settings;
    bad.equation.referenceScales.erase("rho0");
    CHECK_THROWS(C::ResolveReferenceScales(bad.equation));
    bad = settings;
    bad.initialPrimitive[0] = .4;
    CHECK_THROWS(C::ResolvePhysicalStateInputs(bad, equation, {}));
}

#ifdef DNDS_USE_CANTERA
TEST_CASE("Mixture state inputs match Euler total and sensible energies with nonunit scales")
{
    const P::ReferenceScales scales{1.225, 340, 2, 288.15};
    auto pool = std::make_shared<std::vector<P::Chemistry::ChemicalSource>>();
    const std::string path = (std::getenv("DNDS_MECH_PATH") ? std::string(std::getenv("DNDS_MECH_PATH")) + "/" : "") + "h2o2.yaml";
    pool->emplace_back(path, "", scales.U0, scales.rho0);
    auto &source = pool->front();
    P::Chemistry::ChemicalWorkerPool workers(source, 1);
    const auto equation = C::EquationSystem::FromIdealGasMixture(P::Thermodynamics::IdealGasMixture(source, scales), false);
    E::EulerEvaluatorSettings<E::NS_EX_3D>::IdealGasProperty gas;
    gas.rho0 = scales.rho0;
    gas.U0 = scales.U0;
    gas.L0 = scales.L0;
    gas.T0 = scales.T0;
    E::PhysicsProperties<E::NS_EX_3D> legacy(gas);
    legacy.setRANS(E::RANS_None);
    legacy.setChemicalSourcePool(pool);
    E::PhysicsProperties<E::NS_EX_3D>::TU primitive(5 + source.nSpecies() - 1), expected;
    primitive.setZero();
    primitive.head<5>() << .8, .2, -.04, .03, 1400 / scales.T0;
    primitive[5] = .027;
    primitive[8] = .22;
    primitive[10] = .02;
    legacy.primRhoTToConservative(primitive, expected);
    for (const auto *type : {"cons", "consSensible", "primRhoP", "primRhoT", "primTP", "cons_phy", "consSensible_phy", "primRhoP_phy", "primRhoT_phy", "primTP_phy"})
    {
        CAPTURE(type);
        E::PhysicsProperties<E::NS_EX_3D>::TU input;
        legacy.conservativeToStateValueOrigin(expected, input, E::StateValueOriginFromName(type));
        const auto actual = C::ResolvePhysicalState({{"type", type}, {"state", std::vector<double>(input.data(), input.data() + input.size())}}, equation, scales, workers.Acquire());
        for (DNDS::index j = 0; j < expected.size(); ++j)
            CHECK(actual.conservative[j] == doctest::Approx(expected[j]).epsilon(3e-11).scale(1));
        CHECK(actual.temperature == doctest::Approx(1400 / scales.T0).epsilon(3e-11));
    }
}
#endif
