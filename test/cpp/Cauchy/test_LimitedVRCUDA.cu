#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Discretization/CUDA/FiniteVolumeLimitedVRKernels.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeLimitedVRKernels.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "../CFV/ReconstructionFixture.hpp"
#include "CompactManagedStencilFixture.hpp"

namespace C = DNDS::Cauchy;
namespace K = C::CUDA;
namespace R = DNDS::Runtime;
using I = DNDS::index;
using DNDS::real;
static DNDS::MPIInfo mpi;

namespace
{
    void Equal(const R::Array<real> &expected, const R::Array<real> &actual)
    {
        const auto downloaded = actual.To(R::Device::Host());
        const auto a = expected.Read(), b = downloaded.Read();
        REQUIRE(expected.RowCount() == downloaded.RowCount());
        for (I iRow = 0; iRow < expected.RowCount(); ++iRow)
            for (I j = 0; j < expected.RowSize(iRow); ++j)
                CHECK(b.View()(iRow, j) == doctest::Approx(a.View()(iRow, j)).epsilon(2e-12).scale(1));
    }
    void Good(const R::Array<DNDS::NumericalStatus> &array)
    {
        const auto host = array.To(R::Device::Host());
        const auto read = host.Read();
        for (I iRow = 0; iRow < array.RowCount(); ++iRow)
            CHECK(bool(read.View()(iRow, 0)));
    }

}

TEST_CASE("Full-width LVR target and direct sensors agree on CPU and CUDA")
{
    CauchyTest::CompactManagedStencilFixture fixture(CFVTest::CartesianHexes(DNDS::MPIInfo(MPI_COMM_SELF), 2));
    const C::PerfectGas gas;
    const C::ThermalBounds bounds{1e-5, 1e-5, 1e-5};
    const auto device = R::Device::CUDA(0);
    const R::ExecutionContext context{device, nullptr};
    const auto geometry = fixture.metrics.AcquireRead();
    const auto metric = geometry.HostView();
    auto means = fixture.metrics.BuildField(6), gradients = fixture.metrics.BuildField(6 * 3);
    auto target = fixture.metrics.BuildField(6 * 19);
    const auto fill = [&](R::Array<real> &values, R::Array<real> &slopes, I first)
    {
        auto u = values.Write(), gradient = slopes.Write();
        for (I row = 0; row < values.RowCount(); ++row)
        {
            const auto center = metric.GetCellBary(first + row);
            const auto state = gas.Conservative({center[0] < .5 ? 1.8 : 1., center[0] < .5 ? .4 : -.2, .1, 0, center[0] < .5 ? 1.3 : .8});
            for (int j = 0; j < 6; ++j)
            {
                u.View()(row, j) = j < 5 ? state[j] : .1;
                for (int d = 0; d < 3; ++d)
                    gradient.View()(row, 3 * j + d) = d == 0 ? .01 * (j + 1) : 0;
            }
        }
    };
    fill(means.Owned(), gradients.Owned(), 0);
    fill(means.Ghost(), gradients.Ghost(), fixture.metrics.NumCell());
    DNDS::CFV::FVField dm(means.Space(), means.Owned().To(device, context), means.Ghost().To(device, context));
    DNDS::CFV::FVField dg(gradients.Space(), gradients.Owned().To(device, context), gradients.Ghost().To(device, context));
    auto dt = target.Owned().To(device, context);
    const auto deviceMetrics = fixture.metrics.Placed(device, context);
    const auto deviceStencil = fixture.stencil.Placed(deviceMetrics, context);
    const auto hostStencil = fixture.stencil.Read();
    const auto resident = std::make_shared<C::FiniteVolumeStencilStorage::ReadAccess>(deviceStencil.Read(context));
    const auto scalarLayout = R::RowLayout::Dense(means.Owned().RowCount(), 2);
    R::Array<real> scalars(scalarLayout), dsScalar(scalarLayout, device);
    R::Array<DNDS::NumericalStatus> targetStatus(R::RowLayout::Dense(means.Owned().RowCount(), 1), device);
    {
        const auto u = means.Owned().Read(), g = gradients.Owned().Read();
        auto c = target.Owned().Write(), scalar = scalars.Write();
        C::CPU::PrepareLVRScalarsAndTarget(metric.cells, u.View<R::UniformTableRows>(), g.View<R::UniformTableRows>(),
                                           scalar.View<R::UniformTableRows>(), c.View<R::UniformTableRows>(), gas, 3, 19, 0);
        for (I row = 0; row < means.Owned().RowCount(); ++row)
        {
            CHECK(c.View()(row, 5 * 19) == doctest::Approx(.06 * metric.cells(row, 7)));
            CHECK(c.View()(row, 5 * 19 + 3) == 0);
        }
    }
    R::EagerEngine engine;
    const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    R::SubmitArrays(engine, stream, [gas](R::ExecutionContext c, auto cells, auto u, auto g, auto scalars, auto target, auto status)
                    { K::PrepareLVRScalarsAndTarget(c, cells, u, g, scalars, target, gas, 3, 19, 0, status); }, R::ReadAs<R::FixedRows<8>>(deviceMetrics.CellMetrics()), R::ReadAs<R::TableRows>(dm.Owned()), R::ReadAs<R::TableRows>(dg.Owned()), R::WriteAs<R::TableRows>(dsScalar), R::WriteAs<R::TableRows>(dt), R::WriteAs<R::FixedRows<1>>(targetStatus));
    engine.Synchronize();
    Good(targetStatus);
    Equal(target.Owned(), dt);
    Equal(scalars, dsScalar);
    const I faces = fixture.stencil.stencilFaces.RowCount(), cells = fixture.metrics.NumCell();
    R::Array<real> sensors(R::RowLayout::Dense(faces, 3)), branches(R::RowLayout::Dense(cells, 3)), base(R::RowLayout::Dense(cells, 2));
    R::Array<real> ds(sensors.Layout(), device), db(branches.Layout(), device), dbase(base.Layout(), device);
    R::Array<DNDS::NumericalStatus> faceStatus(R::RowLayout::Dense(faces, 1), device), cellStatus(R::RowLayout::Dense(cells, 1), device);
    for (const std::string variant : {"v2", "v3"})
        for (const bool expansion : {false, true})
        {
            auto settings = DNDS::CFV::LimitedVRSettings::ForVariant(variant);
            settings.detectExpansion = expansion;
            const auto parameters = DNDS::CFV::LimitedVRKernelParameters::FromSettings(settings);
            const R::RowSelection selection{R::RowSelectionShape::AllRows(faces), {}};
            {
                const auto u = means.Owned().Read(), g = means.Ghost().Read(), a = gradients.Owned().Read(), b = gradients.Ghost().Read();
                auto s = sensors.Write();
                REQUIRE(bool(C::CPU::BuildLVRFaceSensors(hostStencil.View(), {u.View<R::UniformTableRows>(), g.View<R::UniformTableRows>()},
                                                         {a.View<R::UniformTableRows>(), b.View<R::UniformTableRows>()}, {}, 3, 0, gas, bounds, parameters, s.View<R::UniformTableRows>(), selection)));
            }
            {
                const auto s = sensors.Read();
                auto b = branches.Write(), v = base.Write();
                C::CPU::GatherLVRBranches(hostStencil.View(), s.View<R::UniformTableRows>(), b.View<R::UniformTableRows>(), v.View<R::FixedRows<2>>());
            }
            R::SubmitArrays(engine, stream, [resident, gas, bounds, parameters, selection](R::ExecutionContext c, auto u, auto g, auto a, auto b, auto s, auto status)
                            { K::BuildLVRFaceSensors(c, resident->KernelDescriptor(), {u, g}, {a, b}, {}, 3, 0, gas, bounds, parameters, s, status, selection); }, R::ReadAs<R::TableRows>(dm.Owned()), R::ReadAs<R::TableRows>(dm.Ghost()), R::ReadAs<R::TableRows>(dg.Owned()), R::ReadAs<R::TableRows>(dg.Ghost()), R::WriteAs<R::TableRows>(ds), R::WriteAs<R::FixedRows<1>>(faceStatus));
            R::SubmitArrays(engine, stream, [resident](R::ExecutionContext c, auto s, auto fs, auto b, auto v, auto status)
                            { K::GatherLVRBranches(c, resident->KernelDescriptor(), s, fs, b, v, status); }, R::ReadAs<R::TableRows>(ds), R::ReadAs<R::FixedRows<1>>(faceStatus), R::WriteAs<R::TableRows>(db), R::WriteAs<R::FixedRows<2>>(dbase), R::WriteAs<R::FixedRows<1>>(cellStatus));
            engine.Synchronize();
            Good(faceStatus);
            Good(cellStatus);
            Equal(sensors, ds);
            Equal(branches, db);
            Equal(base, dbase);
        }
    engine.Close();
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
