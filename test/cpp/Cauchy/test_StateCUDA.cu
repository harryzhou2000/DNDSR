#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Discretization/CUDA/FiniteVolumeStateKernels.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeStateKernels.hpp"
#include "Cauchy/Config/SimulationConfig.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "CartesianFixture.hpp"
#include "../CFV/ReconstructionFixture.hpp"
#include "Geom/Quadratures/Quadrature.hpp"

namespace C = DNDS::Cauchy;
namespace K = C::CUDA;
namespace R = DNDS::Runtime;
using I = DNDS::index;
using DNDS::real;
static DNDS::MPIInfo mpi;

TEST_CASE("Queued CUDA owned-mean and CFL reductions match CPU including empty partitions and first failure")
{
    const auto device = R::Device::CUDA(0);
    for (const std::string model : {"none", "SA", "KOmega", "KOmegaSST"})
    {
        C::EquationSettings settings;
        settings.transport = "NewtonianFourier";
        settings.viscosity = .01;
        settings.rans.model = model;
        const auto equation = C::EquationSystem::Resolve(settings);
        const I width = I(equation.Schema().LogicalComponents().size());
        const auto rans = equation.HasRANS() ? C::PrepareRANSStatePolicy(equation, {.1, .2}, 1e5, .01) : C::RANSStateView{};
        const C::AdmissibilityPolicy policy(equation, 1e-8, 1e-8, 0, 1e-8, rans);
        const auto bounds = policy.View();
        for (const I rows : {I(0), I(1), I(1025)})
        {
            R::Array<real> mean(R::RowLayout::Dense(rows, width)), sums(R::RowLayout::Dense(rows, 1)), cells(R::RowLayout::Dense(rows + 3, 8));
            {
                auto u = mean.Write(), s = sums.Write(), m = cells.Write();
                for (I iCell = 0; iCell < rows; ++iCell)
                {
                    const auto flow = equation.Gas().Conservative({1 + .001 * iCell, .2, .1, -.1, 1.1});
                    for (I j = 0; j < width; ++j)
                        u.View()(iCell, j) = j < 5 ? flow[j] : .1;
                    s.View()(iCell, 0) = 2 + .07 * iCell;
                    m.View()(iCell, 0) = .1 + .003 * iCell;
                }
            }
            auto dm = mean.To(device), ds = sums.To(device), dc = cells.To(device);
            R::Array<K::StateReductionValue> scratch(R::RowLayout::Dense(K::StateReductionScratchRows(rows), 1), device);
            R::Array<K::StateReductionValue> valid(R::RowLayout::Dense(1, 1), device), step(valid.Layout(), device);
            R::EagerEngine engine;
            const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
            const auto compare = [&]
            {
                bool expectedValid;
                DNDS::NumericalResult<real> expectedStep{real(0)};
                {
                    const auto u = mean.Read(), s = sums.Read(), m = cells.Read();
                    std::vector<DNDS::Algebra::DofComponentColumn<const real>> columns;
                    for (I j = 0; j < width; ++j)
                        columns.push_back({u.View<R::UniformTableRows>(), {}, j});
                    const DNDS::Algebra::DofComponentLayout<const real> components(std::move(columns), rows, 0);
                    expectedValid = C::CPU::AdmissibleMeans(components.View(), bounds);
                    DNDS::CFV::FiniteVolumeHostView metric;
                    metric.cells = m.View<R::FixedRows<8>, false>();
                    expectedStep = C::CPU::MinimumStableStep(s.View<R::UniformTableRows>(), metric, 3.5);
                }
                R::SubmitArrays(engine, stream, [bounds](R::ExecutionContext c, auto u, auto work, auto result)
                                { K::AdmissibleMeans(c, u, bounds, work, result); }, R::ReadAs<R::TableRows>(dm), R::WriteAs<R::FixedRows<1>>(scratch), R::WriteAs<R::FixedRows<1>>(valid));
                // Reuse scratch on the same stream without a host wait.
                R::SubmitArrays(engine, stream, [](R::ExecutionContext c, auto s, auto m, auto work, auto result)
                                { K::MinimumStableStep(c, s, m, 3.5, work, result); }, R::ReadAs<R::TableRows>(ds), R::ReadAs<R::FixedRows<8>>(dc), R::WriteAs<R::FixedRows<1>>(scratch), R::WriteAs<R::FixedRows<1>>(step));
                engine.Synchronize();
                const auto hv = valid.To(R::Device::Host()), hs = step.To(R::Device::Host());
                const auto v = hv.Read(), s = hs.Read();
                CHECK(v.View()(0, 0).value == real(expectedValid));
                CHECK((v.View()(0, 0).firstInvalid < 0) == expectedValid);
                CHECK((s.View()(0, 0).firstInvalid < 0) == bool(expectedStep));
                if (expectedStep)
                    CHECK(s.View()(0, 0).value == doctest::Approx(expectedStep.value).epsilon(2e-14));
                else
                    CHECK(s.View()(0, 0).firstInvalid == expectedStep.status.row);
            };
            compare();
            if (rows > 1)
            {
                mean.Write().At(rows / 2, 0) = -1;
                sums.Write().At(rows - 1, 0) = 0;
                dm.CopyFrom(mean, stream.Context()).Wait();
                ds.CopyFrom(sums, stream.Context()).Wait();
                compare();
            }
        }
    }
}

TEST_CASE("CUDA analytic error uses the CPU oracle quadrature and fixed-tree rank-local reductions")
{
    for (const int dimension : {2, 3})
    {
        const auto mesh = dimension == 2 ? CauchyTest::PeriodicQuads(mpi, 17) : CFVTest::CartesianHexes(mpi, 3);
        const auto metrics = DNDS::CFV::FiniteVolumePlan::ConstructMetrics(mesh);
        const auto geometry = metrics.AcquireRead();
        const auto metric = geometry.HostView();
        const auto meshRead = mesh->AcquireRead(DNDS::Geom::MeshReadRequirements::CellFlux());
        const auto view = meshRead.HostView();
        const I rows = metrics.NumCell();
        std::vector<I> starts(size_t(rows + 1), 0);
        for (I iCell = 0; iCell < rows; ++iCell)
            starts[size_t(iCell + 1)] = starts[size_t(iCell)] + DNDS::Geom::Elem::Quadrature(
                                                                    view.GetCellElement(iCell), DNDS::Geom::Elem::INT_ORDER_MAX)
                                                                    .GetNumPoints();
        R::Array<I> offsets(R::RowLayout::Dense(rows + 1, 1));
        R::Array<real> points(R::RowLayout::Dense(starts.back(), 4)), cells(R::RowLayout::Dense(rows, 8));
        {
            auto o = offsets.Write();
            auto p = points.Write(), c = cells.Write();
            for (I iRow = 0; iRow <= rows; ++iRow)
                o.View()(iRow, 0) = starts[size_t(iRow)];
            for (I iCell = 0; iCell < rows; ++iCell)
            {
                for (int j = 0; j < 8; ++j)
                    c.View()(iCell, j) = metric.cells(iCell, j);
                DNDS::Geom::tSmallCoords coords;
                view.GetCoordsOnCell(iCell, coords);
                const auto element = view.GetCellElement(iCell);
                const DNDS::Geom::Elem::Quadrature quadrature(element, DNDS::Geom::Elem::INT_ORDER_MAX);
                for (int iQ = 0; iQ < quadrature.GetNumPoints(); ++iQ)
                {
                    const auto &shape = DNDS::Geom::Elem::GetCachedShapeAtQuadrature(element.type, DNDS::Geom::Elem::INT_ORDER_MAX, iQ);
                    const DNDS::Geom::tPoint point = DNDS::Geom::Elem::PPhysicsCoordD01Nj(coords, shape);
                    for (int d = 0; d < 3; ++d)
                        p.View()(starts[size_t(iCell)] + iQ, d) = point[d];
                    p.View()(starts[size_t(iCell)] + iQ, 3) = DNDS::Geom::Elem::CellJacobianDet(dimension, coords, shape) * quadrature.GetWeight(iQ);
                }
            }
        }
        const auto device = R::Device::CUDA(0);
        auto doffsets = offsets.To(device);
        auto dpoints = points.To(device), dcells = cells.To(device);
        C::PerfectGas gas;
        C::RigidMotionSettings motion;
        motion.angularVelocity = {0, 0, .2};
        motion.translationVelocity = {.1, -.2, 0};
        const C::RigidMotion moving(motion, dimension);
        const real time = .35;
        const auto frame = moving.At(time);
        const C::FlowState primitive{1, .3, -.2, .1, 1};
        for (const std::string kind : {"constant", "isentropicVortex"})
        {
            C::FlowProfileSettings profile;
            profile.kind = kind;
            profile.strength = .8;
            profile.center = {.4, .6, 0};
            profile.period = 1;
            // Extra scalar components are deliberately unrelated to the flow
            // oracle; the kernel must use only the first five components.
            R::Array<real> means(R::RowLayout::Dense(rows, 7));
            {
                auto write = means.Write();
                REQUIRE(bool(C::CPU::InitializeFlowProfile(view, {write.View<R::UniformTableRows>(), 0, 5}, profile, primitive, gas, moving.At(0))));
                for (I iCell = 0; iCell < rows; ++iCell)
                {
                    write.View()(iCell, 0) += .001 * (iCell + 1);
                    write.View()(iCell, 5) = 100;
                    write.View()(iCell, 6) = -100;
                }
            }
            const auto read = means.Read();
            const auto expected = C::CPU::MeasureFlowError(view, metric, {read.View<R::UniformTableRows>(), 0, 5}, profile, primitive, gas, time, frame);
            REQUIRE(bool(expected));
            auto dmeans = means.To(device);
            R::Array<K::FlowErrorReduction> scratch(R::RowLayout::Dense(K::StateReductionScratchRows(rows), 1), device);
            R::Array<K::FlowErrorReduction> output(R::RowLayout::Dense(1, 1), device), repeat(output.Layout(), device);
            R::EagerEngine engine;
            const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
            const auto submit = [&](R::Array<K::FlowErrorReduction> &result)
            {
                R::SubmitArrays(engine, stream, [parameters = C::PrepareFlowProfile(profile), primitive, gas, frame = frame.KernelParameters(), time](R::ExecutionContext context, auto offsets, auto points, auto cells, auto means, auto scratch, auto output)
                                { K::MeasureFlowError(context, {offsets, points, cells}, means, parameters, primitive, gas, frame, time, scratch, output); }, R::ReadAs<R::FixedRows<1>>(doffsets), R::ReadAs<R::FixedRows<4>>(dpoints), R::ReadAs<R::FixedRows<8>>(dcells), R::ReadAs<R::TableRows>(dmeans), R::WriteAs<R::FixedRows<1>>(scratch), R::WriteAs<R::FixedRows<1>>(result));
            };
            submit(output);
            submit(repeat); // no host visibility or scratch reallocation between submissions
            engine.Synchronize();
            const auto host = output.To(R::Device::Host()), again = repeat.To(R::Device::Host());
            const auto a = host.Read(), b = again.Read();
            const auto actual = a.View()(0, 0), repeated = b.View()(0, 0);
            REQUIRE(bool(actual.status));
            CHECK(actual.volume == doctest::Approx(expected.value.volume).epsilon(2e-13));
            CHECK(actual.volume == repeated.volume);
            for (int norm = 0; norm < 3; ++norm)
                for (int j = 0; j < 5; ++j)
                {
                    CHECK(actual.errors[norm][j] == doctest::Approx(expected.value.errors[norm][j]).epsilon(2e-11).scale(1));
                    CHECK(actual.errors[norm][j] == repeated.errors[norm][j]);
                }
        }
    }
}

int main(int argc, char **argv)
{
    DNDS::MPI::Init_thread(&argc, &argv);
    mpi = DNDS::MPIInfo(MPI_COMM_WORLD);
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const int result = context.run();
    DNDS::MPI::Finalize();
    return result;
}
