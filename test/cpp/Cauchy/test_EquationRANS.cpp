#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Cauchy/Physics/EquationRANS.hpp"
#include "Cauchy/State/EquationDofs.hpp"
#include "Cauchy/Config/SimulationConfig.hpp"

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
namespace A = DNDS::Algebra;
using I = DNDS::index;

namespace
{
    C::EquationSystem Equation(const char *model)
    {
        C::EquationSettings settings;
        settings.transport = "NewtonianFourier";
        settings.viscosity = .002;
        DNDS::Physics::RANS::ModelSettings rans;
        rans.model = model;
        return C::EquationSystem::Resolve(settings).WithRANS(rans);
    }
    R::RowLayout Layout(I width, int layout)
    {
        if (layout == 1)
            return R::RowLayout::ColMajor(0, width, {16, 1});
        if (layout == 2)
            return R::RowLayout::TiledColMajor(0, width, 32);
        return R::RowLayout::Dense(0, width, width + 2);
    }
}

TEST_CASE("RANS model and point gathering follow logical blocks across split and coupled layouts")
{
    for (const auto *name : {"SA", "KOmegaWilcox", "KOmegaSST", "RealizableKE"})
    {
        const auto equation = Equation(name);
        const auto &model = equation.RANS();
        const I n = model.ComponentNumber();
        REQUIRE(equation.Schema().blocks.size() == 2);
        CHECK(equation.Schema().blocks[1].name == "rans");
        CHECK(equation.Schema().ResolveComponents({model.ComponentNames()[0]}) == std::vector<I>{5});
        auto policy = C::PrepareRANSStatePolicy(equation, {.01, 3});
        for (int layout = 0; layout < 3; ++layout)
            for (bool split : {false, true})
            {
                const A::DofSpace space(35, 0, "RANS point layouts");
                A::DofBundle fields(space);
                std::vector<C::EquationDofBinding> bindings;
                if (split)
                {
                    fields.Add("turb", A::DofField<double>(space, Layout(n, layout)));
                    fields.Add("fluid", A::DofField<double>(space, Layout(5, layout)));
                    bindings = {{"flow", "fluid", 0}, {"rans", "turb", 0}};
                }
                else
                {
                    fields.Add("coupled", A::DofField<double>(space, Layout(5 + n, layout)));
                    // RANS physically precedes flow, contrary to logical order.
                    bindings = {{"flow", "coupled", n}, {"rans", "coupled", 0}};
                }
                C::EquationDofs state(equation.Schema(), std::move(fields), std::move(bindings));
                equation.ValidateStateSchema(state);
                {
                    auto write = state.WriteOwned();
                    auto flow = write.Block("flow"), scalars = write.Block("rans");
                    for (I iCell = 0; iCell < 35; ++iCell)
                    {
                        for (I j = 0; j < 5; ++j)
                            flow(iCell, j) = 2 + .01 * iCell + .1 * j;
                        for (I j = 0; j < n; ++j)
                            scalars(iCell, j) = .02 + .001 * iCell + .3 * j;
                    }
                }
                auto read = state.ReadOwned();
                const auto flow = read.Block("flow"), scalars = read.Block("rans");
                Eigen::Matrix<double, 5, 3> flowGradient;
                Eigen::Matrix<double, 2, 3> scalarGradient;
                for (int j = 0; j < 5; ++j)
                    for (int d = 0; d < 3; ++d)
                        flowGradient(j, d) = .01 * (1 + j - 2 * d);
                scalarGradient << .1, -.2, .3, -.1, .05, .08;
                for (I iCell = 0; iCell < 35; ++iCell)
                {
                    const auto point = C::PrepareRANSPoint(policy, flow[iCell], scalars[iCell], flowGradient, scalarGradient,
                                                           .002, 3.5, 1.4, .1, .03);
                    for (int j = 0; j < 5; ++j)
                    {
                        CHECK(point.state[j] == 2 + .01 * iCell + .1 * j);
                        for (int d = 0; d < 3; ++d)
                            CHECK(point.gradient(d, j) == flowGradient(j, d));
                    }
                    for (int j = 0; j < n; ++j)
                    {
                        CHECK(point.state[5 + j] == .02 + .001 * iCell + .3 * j);
                        for (int d = 0; d < 3; ++d)
                            CHECK(point.gradient(d, 5 + j) == scalarGradient(j, d));
                    }
                    if (n == 1)
                    {
                        CHECK(point.state[6] == 0);
                        CHECK(point.gradient.col(6).isZero());
                    }
                    CHECK(std::isfinite(model.Source(point)[0]));
                    CHECK(std::isfinite(model.Transport(point).viscosity));
                }
            }
    }
    auto base = C::EquationSystem::Resolve(C::EquationSettings{});
    DNDS::Physics::RANS::ModelSettings rans;
    rans.model = "SA";
    CHECK_THROWS(base.WithRANS(rans)); // Inviscid transport cannot host RANS.
    CHECK_FALSE(base.HasRANS());
    CHECK_THROWS(Equation("SA").WithRANS(rans));
}

TEST_CASE("RANS scalar repairs retain the Euler componentwise and exponential rules")
{
    for (const auto *name : {"SA", "KOmegaWilcox", "KOmegaSST", "RealizableKE"})
    {
        const auto equation = Equation(name);
        const auto policy = C::PrepareRANSStatePolicy(equation, {.2, 4.}, 10., .02);
        const bool sa = policy.components == 1;
        const std::array<double, 2> mean{.2, 1.};
        std::array<double, 2> sample{-.1, 2.};
        policy.RepairReconstruction(mean, sample);
#ifdef USE_NS_SA_NUT_REDUCED_ORDER
        CHECK(sample[0] == mean[0]);
#else
        CHECK(sample[0] == (sa ? -.1 : mean[0]));
#endif
        CHECK(sample[1] == 2.); // No global contraction of other components.
        std::array<double, 2> delta{-3., -2.};
        policy.RepairIncrement(mean, delta);
        for (int j = 0; j < policy.components; ++j)
        {
#ifdef USE_NS_SA_ALLOW_NEGATIVE_MEAN
            if (sa)
            {
                CHECK(delta[j] == -3.);
                continue;
            }
#endif
            const double raw = mean[j] * std::exp((j ? -2. : -3.) / (mean[j] + 1e-6));
            const double expected = sa ? std::max(1e-6, raw) : raw;
            CHECK(mean[j] + delta[j] == doctest::Approx(expected).epsilon(1e-11));
        }
        std::array<double, 2> next{100., .001};
        policy.RepairMean(2., next);
        CHECK(next[0] == (sa ? 20. : 100.));
        const bool omega = equation.RANS().Kind() == DNDS::Physics::RANS::ModelKind::KOmegaWilcox ||
                           equation.RANS().Kind() == DNDS::Physics::RANS::ModelKind::KOmegaSST;
        CHECK(next[1] == (omega ? .08 : .001));
    }
}

TEST_CASE("RANS reconstruction ratios retain distinct reference and compression thresholds")
{
    struct Samples
    {
        std::array<std::array<double, 3>, 2> values;
        double operator()(I q, I j) const { return values[q][j]; }
    };
    const Samples base{std::array{std::array{9., .2, .4}, std::array{9., .2, .4}}};
    const Samples higher{std::array{std::array{9., -.2, .5}, std::array{9., .3, -.4}}};
    for (const auto *name : {"SA", "KOmegaWilcox", "KOmegaSST", "RealizableKE"})
    {
        const auto equation = Equation(name);
        C::RANSReconstructionView policy{equation.RANS().Kind(), equation.RANS().ComponentNumber(), .01, {.02, .04}};
        CHECK(policy.ReferenceAdmissible(std::array<double, 2>{.015, .03}));
        CHECK_FALSE(policy.ReferenceAdmissible(std::array<double, 2>{.005, .03}));
        const auto result = policy.Compute(base, higher, 1, 0, 2);
        if (policy.components == 2)
        {
            // (.2-.02)/(.2-(-.2)) = (.4-.04)/(.4-(-.4)) = .45.
            // Euler damps both entire scalar coefficient columns by .9*.45.
            CHECK(result.coefficientScale == doctest::Approx(.405).epsilon(1e-14));
            CHECK(result.flowDensityFraction == 1);
        }
        else
        {
            CHECK(result.coefficientScale == 1);
#ifdef USE_NS_SA_NUT_REDUCED_ORDER
            CHECK(result.flowDensityFraction == doctest::Approx(.45).epsilon(1e-14));
#else
            CHECK(result.flowDensityFraction == 1);
#endif
        }
        const auto unchanged = policy.Compute(base, base, 1, 0, 2);
        CHECK(unchanged.coefficientScale == 1);
        CHECK(unchanged.flowDensityFraction == 1);
    }
}

TEST_CASE("Named RANS initialization and numerical reference resolve against the model")
{
    C::RANSStateSettings settings;
    settings.saUpperLimit = 12;
    const auto sa = Equation("SA");
    CHECK(C::ResolveRANSValues(sa, {{"rhoNuTilde", .02}})[0] == .02);
    CHECK(C::ResolveRANSStatePolicy(sa, settings).saUpperLimit == 12);
    CHECK_THROWS(C::ResolveRANSValues(sa, {{"rhoK", .02}}));
    for (const auto *name : {"KOmegaWilcox", "KOmegaSST"})
    {
        const auto equation = Equation(name);
        settings.referenceConservative.clear();
        CHECK_THROWS(C::ResolveRANSStatePolicy(equation, settings));
        settings.referenceConservative = {{"rhoK", .1}, {"rhoOmega", 4}};
        CHECK(C::ResolveRANSStatePolicy(equation, settings).omegaLowerLimit == .04);
        CHECK_THROWS(C::ResolveRANSReconstructionPolicy(equation, settings, 1e-12));
        settings.referenceConservative["rho"] = .25;
        const auto reconstruction = C::ResolveRANSReconstructionPolicy(equation, settings, 1e-12);
        CHECK(reconstruction.compressionFloors[0] == doctest::Approx(DNDS::smallReal * .4).scale(1e-30));
        CHECK(reconstruction.compressionFloors[1] == doctest::Approx(DNDS::smallReal * 16).scale(1e-30));
        CHECK(C::ResolveRANSStatePolicy(equation, settings).omegaLowerLimit == .04);
        const auto roundTrip = C::Json(settings).get<C::RANSStateSettings>();
        CHECK(roundTrip.referenceConservative == settings.referenceConservative);
        settings.referenceConservative["rho"] = 0;
        CHECK_THROWS(C::ResolveRANSReconstructionPolicy(equation, settings, 1e-12));
    }
}

TEST_CASE("RANS no-slip targets retain conservative density and viscosity conventions")
{
    DNDS::Physics::RANS::Point mean;
    mean.state[0] = 2;
    mean.state[5] = .06;
    mean.wallDistance = .02;
    for (double mu : {.001, .004}) // Mean-temperature and wall-temperature producers.
    {
        mean.molecularViscosity = mu;
        for (const auto *name : {"SA", "KOmegaWilcox", "KOmegaSST", "RealizableKE"})
        {
            const auto equation = Equation(name);
            const auto wall = C::RANSWallTarget(equation.RANS(), mean, 1.5);
            CHECK(wall[0] == 0);
            const double expected = std::string(name) == "SA" ? 0 : std::string(name) == "RealizableKE" ? 2 * mu / 2 * (.06 / 2) / (.02 * .02) * 1.5
                                                                                                        : 800 * mu / (.02 * .02);
            CHECK(wall[1] == doctest::Approx(expected));
        }
    }
}

TEST_CASE("Two-equation isotropic stress and its conservative derivative preserve pressure work")
{
    const C::FlowState state{1.7, .6, -.4, .2, 5.};
    const C::Vector3 normal{.6, -.8, 0};
    const double rhoK = .09, eps = 1e-6;
    const auto flux = C::TurbulentNormalFlux(state, rhoK, normal);
    CHECK(flux[0] == 0);
    CHECK(flux[1] == doctest::Approx(.06 * .6));
    CHECK(flux[2] == doctest::Approx(-.06 * .8));
    CHECK(flux[4] == doctest::Approx(.06 * (.6 * .6 + .4 * .8) / 1.7));
    for (int component = 0; component < 7; ++component)
    {
        C::FlowState direction{}, plus = state, minus = state;
        if (component < 5)
        {
            direction[component] = 1;
            plus[component] += eps;
            minus[component] -= eps;
        }
        const double deltaK = component == 5 ? 1 : 0;
        const auto derivative = C::TurbulentNormalFluxDerivative(state, direction, rhoK, deltaK, normal);
        const auto a = C::TurbulentNormalFlux(plus, rhoK + eps * deltaK, normal);
        const auto b = C::TurbulentNormalFlux(minus, rhoK - eps * deltaK, normal);
        for (int j = 0; j < 5; ++j)
            CHECK(derivative[j] == doctest::Approx((a[j] - b[j]) / (2 * eps)).epsilon(2e-9).scale(1e-9));
    }
}
