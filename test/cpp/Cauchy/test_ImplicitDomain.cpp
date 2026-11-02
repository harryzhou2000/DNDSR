#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Core/Domain.hpp"
#include "CartesianFixture.hpp"
#include <optional>

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
static DNDS::MPIInfo mpi;
namespace
{
    std::array<double, 5> Integral(const C::Domain &domain)
    {
        std::array<double, 5> value{};
        const auto read = domain.Flow().Owned().Read();
        const auto data = read.View<R::UniformTableRows>();
        for (DNDS::index iCell = 0; iCell < data.RowCount(); ++iCell)
            for (int j = 0; j < 5; ++j)
                value[j] += data(iCell, j) * domain.Discretization().Metrics().GetCellVol(iCell);
        MPI_Allreduce(MPI_IN_PLACE, value.data(), 5, MPI_DOUBLE, MPI_SUM, mpi.comm);
        return value;
    }
    void DensityWave(C::Domain &domain)
    {
        constexpr double pi = 3.14159265358979323846;
        auto write = domain.Flow().Owned().Write();
        const auto values = write.View<R::UniformTableRows>();
        for (DNDS::index iCell = 0; iCell < values.RowCount(); ++iCell)
        {
            const auto point = domain.Discretization().Metrics().GetCellCent(iCell);
            const double rho = 1 + .05 * std::sin(2 * pi * (point[0] + point[1]));
            const auto state = domain.Equation().Gas().Conservative({rho, .4, .2, .1, 1});
            for (int j = 0; j < 5; ++j)
                values(iCell, j) = state[j];
        }
    }
}
TEST_CASE("Native physical implicit methods preserve uniform periodic flow and transactional commit")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    for (const std::string method : {"bdf1", "bdf2", "trapz", "esdirk2", "esdirk3", "esdirk4", "ditr"})
    {
        C::DomainSettings settings;
        settings.numerics.reconstruction = "constant";
        settings.implicit.minIterations = 0;
        settings.initialPrimitive = {1.1, .3, -.2, .4, 1.2};
        settings.execution.hostThreads = method == "esdirk4" ? 2 : 0;
        C::TimeSettings time;
        time.method = method;
        C::Domain domain("uniform", mesh, settings, time);
        const auto initial = Integral(domain);
        domain.Advance(.01);
        domain.Advance(.02);
        CHECK(domain.Time() == doctest::Approx(.03));
        CHECK(domain.Epoch().acceptedStep == 2);
        for (const auto &report : domain.LastImplicitStages())
            CHECK(report.converged);
        domain.PrepareStep(.01);
        std::optional<R::Array<double>::ReadLease> held;
        if (mpi.rank == 0)
            held.emplace(domain.Flow().Owned().Read());
        CHECK_THROWS(domain.CommitStep());
        CHECK(domain.Time() == doctest::Approx(.03));
        held.reset();
        domain.CommitStep();
        CHECK(domain.Time() == doctest::Approx(.04));
        CHECK(domain.Epoch().acceptedStep == 3);
        const auto final = Integral(domain);
        for (int j = 0; j < 5; ++j)
            CHECK(final[j] == doctest::Approx(initial[j]).epsilon(5e-14));
    }
}
TEST_CASE("P3 implicit periodic transport conserves flow and discard restores the VR warm guess")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 6);
    for (const std::string method : {"esdirk4", "ditr"})
    {
        C::DomainSettings settings;
        settings.numerics.reconstruction = "variational";
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.reconstructionIterations = 120;
        settings.numerics.reconstructionTolerance = 1e-10;
        settings.numerics.faceFallback = false;
        settings.implicit.relativeTolerance = 1e-9;
        settings.implicit.absoluteTolerance = 1e-11;
        settings.implicit.linear.relativeTolerance = 1e-3;
        settings.implicit.minIterations = 0;
        C::TimeSettings time;
        time.method = method;
        C::Domain retried("retry", mesh, settings, time), direct("direct", mesh, settings, time);
        DensityWave(retried);
        DensityWave(direct);
        const auto initial = Integral(retried);
        retried.PrepareStep(.005);
        retried.DiscardStep();
        CHECK(retried.Time() == 0);
        CHECK(retried.Epoch().acceptedStep == 0);
        retried.Advance(.005);
        direct.Advance(.005);
        CHECK(retried.Time() == .005);
        CHECK(retried.LastFaceFallbacks() == 0);
        {
            const auto a = retried.Flow().Owned().Read(), b = direct.Flow().Owned().Read();
            for (DNDS::index iCell = 0; iCell < a.View().RowCount(); ++iCell)
                for (int j = 0; j < 5; ++j)
                    CHECK(a.View()(iCell, j) == b.View()(iCell, j));
        }
        const auto final = Integral(retried);
        for (int j = 0; j < 5; ++j)
            CHECK(final[j] == doctest::Approx(initial[j]).epsilon(2e-11));
        if (method == "ditr")
        {
            CHECK(retried.LastCoupledIterations() > 0);
            CHECK(retried.LastCoupledTargetRatio() <= 1);
            CHECK(retried.LastImplicitStages().size() == size_t(2 * retried.LastCoupledIterations()));
        }
        else
            CHECK(retried.LastImplicitStages().size() == 5);
        for (const auto &report : retried.LastImplicitStages())
        {
            CHECK(report.converged);
            CHECK(report.iterations > 0);
            CHECK(report.reconstructionIterations > 0);
            CHECK(report.reconstructionIterations <= settings.numerics.reconstructionIterations);
            CHECK(report.reconstructionRelativeUpdate <= settings.numerics.reconstructionTolerance);
        }
        CHECK(retried.Discretization().Admissible(retried.Flow()));
    }
}
TEST_CASE("DITR coupled cap rejects the physical trial even when frozen stage solves converge")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    C::DomainSettings settings;
    settings.numerics.reconstruction = "constant";
    settings.implicit.minIterations = 0;
    settings.implicit.relativeTolerance = 1e-10;
    settings.implicit.absoluteTolerance = 1e-12;
    C::TimeSettings time;
    time.method = "ditr";
    time.ditr.iterations = 1;
    time.ditr.relativeTolerance = 1e-10;
    time.ditr.absoluteTolerance = 1e-12;
    C::Domain domain("capped", mesh, settings, time);
    DensityWave(domain);
    const auto before = Integral(domain);
    const auto version = domain.Flow().Owned().Version();
    CHECK_THROWS_WITH_AS(domain.PrepareStep(.01), doctest::Contains("last_residual"), std::exception);
    CHECK(domain.Time() == 0);
    CHECK(domain.Epoch().acceptedStep == 0);
    CHECK(domain.Flow().Owned().Version() == version);
    CHECK(Integral(domain) == before);
    CHECK(domain.LastCoupledIterations() == 1);
    CHECK(domain.LastCoupledTargetRatio() > 1);
    REQUIRE(domain.LastImplicitStages().size() == 2);
    for (const auto &inner : domain.LastImplicitStages())
        CHECK(inner.converged);
    CHECK_THROWS(domain.CommitStep());
}
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    mpi.setWorld();
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const auto result = context.run();
    MPI_Finalize();
    return result;
}
