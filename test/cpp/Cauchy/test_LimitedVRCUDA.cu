#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Cauchy/Discretization/CUDA/FiniteVolumeLimitedVRKernels.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeLimitedVRKernels.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"

namespace C = DNDS::Cauchy;
namespace K = C::CUDA;
namespace R = DNDS::Runtime;
using I = DNDS::index;
using DNDS::real;
namespace
{
    template <class T, class F>
    R::Array<T> Table(I rows, I width, F fill)
    {
        R::Array<T> result(R::RowLayout::Dense(rows, width));
        auto write = result.Write();
        for (I iRow = 0; iRow < rows; ++iRow)
            for (I j = 0; j < width; ++j)
                write.View()(iRow, j) = fill(iRow, j);
        return result;
    }
    void Equal(const R::Array<real> &expected, const R::Array<real> &actual)
    {
        const auto downloaded = actual.To(R::Device::Host());
        const auto a = expected.Read(), b = downloaded.Read();
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

TEST_CASE("Physical LVR CUDA sensors gather and publication preserve the CPU branch gates")
{
    const C::PerfectGas gas;
    const C::ThermalBounds bounds{1e-8, 1e-8, 1e-8};
    const auto device = R::Device::CUDA(0);
    const auto context = R::ExecutionContext::CUDA(0);
    const std::array<C::FlowState, 3> states{
        gas.Conservative({1., .4, .2, 0, 1.}),
        gas.Conservative({1.8, -.2, .1, 0, 1.3}),
        gas.Conservative({.8, .1, -.3, 0, .9})};
    auto means = Table<real>(2, 5, [&](I iRow, I j)
                             { return states[iRow][j]; });
    auto ghosts = Table<real>(1, 5, [&](I, I j)
                              { return states[2][j]; });
    auto faces = Table<C::LimitedVRFace>(3, 1, [](I iFace, I)
                                         {
        C::LimitedVRFace face{};
        face.cell = iFace == 2 ? 1 : 0;
        face.other = iFace == 0 ? 1 : iFace == 1 ? 2 : 1;
        face.normal = {1, 0, 0};
        face.rotation = iFace == 2 ? std::array<real, 9>{0, 1, 0, -1, 0, 0, 0, 0, 1}
                                  : std::array<real, 9>{1, 0, 0, 0, 1, 0, 0, 0, 1};
        face.changesFrame = iFace == 2;
        return face; });
    auto samples = Table<real>(6, 5, [&](I iRow, I j)
                               {
        const std::array<int, 6> cell{0, 1, 0, 2, 1, 1};
        return states[cell[iRow]][j] * (iRow % 2 ? 1.07 : .93); });
    // Two owned receivers: [face0, face1], [face0, face2, face2].
    // The self-periodic face's repeated incidence is intentional.
    auto offsets = Table<I>(3, 1, [](I iRow, I)
                            { return std::array<I, 3>{0, 2, 5}[iRow]; });
    auto indices = Table<I>(5, 1, [](I iRow, I)
                            { return std::array<I, 5>{0, 1, 0, 2, 2}[iRow]; });
    auto dm = means.To(device), dg = ghosts.To(device), ds = samples.To(device);
    auto df = faces.To(device);
    auto dOffsets = offsets.To(device), dIndices = indices.To(device);
    R::Array<real> branches(R::RowLayout::Dense(2, 3)), base(R::RowLayout::Dense(2, 2));
    R::Array<real> history(R::RowLayout::Dense(2, 2)), penalty(R::RowLayout::Dense(2, 1)), scalar(R::RowLayout::Dense(2, 2));
    R::Array<real> db(branches.Layout(), device), dbase(base.Layout(), device), dh(history.Layout(), device), da(penalty.Layout(), device), dscalar(scalar.Layout(), device);
    R::Array<real> sensors(R::RowLayout::Dense(3, 3), device);
    R::Array<real> cpuSensors(R::RowLayout::Dense(3, 3));
    R::Array<DNDS::NumericalStatus> faceStatus(R::RowLayout::Dense(3, 1), device), cellStatus(R::RowLayout::Dense(2, 1), device), scalarStatus(cellStatus.Layout(), device);
    for (const std::string variant : {"v2", "v3"})
        for (const bool expansion : {false, true})
        {
            auto settings = DNDS::CFV::LimitedVRSettings::ForVariant(variant);
            settings.detectExpansion = expansion;
            {
                const auto u = means.Read(), g = ghosts.Read(), q = samples.Read();
                const auto f = faces.Read();
                auto b = branches.Write(), v = base.Write(), s = scalar.Write();
                REQUIRE(bool(C::CPU::BuildLVRBranches(q.View<R::UniformTableRows>(), {u.View<R::UniformTableRows>(), g.View<R::UniformTableRows>()},
                                                      f.View<R::FixedRows<1>>(), gas, bounds, settings, b.View<R::UniformTableRows>(), v.View<R::FixedRows<2>>())));
                std::array<real, 6> serial{};
                for (I iCell = 0; iCell < 2; ++iCell)
                    for (I j = 0; j < 3; ++j)
                        serial[size_t(iCell * 3 + j)] = b.View()(iCell, j);
                {
                    auto sensor = cpuSensors.Write();
                    REQUIRE(bool(C::CPU::BuildLVRFaceSensors(q.View<R::UniformTableRows>(), {u.View<R::UniformTableRows>(), g.View<R::UniformTableRows>()},
                                                             f.View<R::FixedRows<1>>(), gas, bounds, settings, sensor.View<R::UniformTableRows>())));
                }
                const auto sensor = cpuSensors.Read();
                const auto off = offsets.Read(), ind = indices.Read();
                C::CPU::GatherLVRBranches(off.View<R::FixedRows<1>>(), ind.View<R::FixedRows<1>>(), sensor.View<R::UniformTableRows>(),
                                          b.View<R::UniformTableRows>(), v.View<R::FixedRows<2>>());
                for (I iCell = 0; iCell < 2; ++iCell)
                    for (I j = 0; j < 3; ++j)
                        CHECK(b.View()(iCell, j) == serial[size_t(iCell * 3 + j)]);
                C::CPU::FillLVRScalars(u.View<R::UniformTableRows>(), s.View<R::UniformTableRows>(), gas);
            }
            {
                auto b = branches.Write();
                const auto v = base.Read();
                C::CPU::MergeLVRBase(v.View<R::FixedRows<2>>(), b.View<R::UniformTableRows>(), 0);
            }
            {
                const auto b = branches.Read();
                auto h = history.Write(), a = penalty.Write();
                C::CPU::PublishLVRPenalty(b.View<R::UniformTableRows>(), h.View<R::FixedRows<2>>(), a.View<R::FixedRows<1>>());
            }
            R::EagerEngine engine;
            const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
            const auto parameters = DNDS::CFV::LimitedVRKernelParameters::FromSettings(settings);
            R::SubmitArrays(engine, stream, [gas, bounds, parameters](R::ExecutionContext c, auto q, auto u, auto g, auto f, auto s, auto status)
                            { K::BuildLVRFaceSensors(c, q, {u, g}, f, gas, bounds, parameters, s, status); }, R::ReadAs<R::TableRows>(ds), R::ReadAs<R::TableRows>(dm), R::ReadAs<R::TableRows>(dg), R::ReadAs<R::FixedRows<1>>(df), R::WriteAs<R::TableRows>(sensors), R::WriteAs<R::FixedRows<1>>(faceStatus));
            R::SubmitArrays(engine, stream, [](R::ExecutionContext c, auto o, auto i, auto s, auto fs, auto b, auto v, auto status)
                            { K::GatherLVRBranches(c, o, i, s, fs, b, v, status); }, R::ReadAs<R::FixedRows<1>>(dOffsets), R::ReadAs<R::FixedRows<1>>(dIndices), R::ReadAs<R::TableRows>(sensors), R::ReadAs<R::FixedRows<1>>(faceStatus), R::WriteAs<R::TableRows>(db), R::WriteAs<R::FixedRows<2>>(dbase), R::WriteAs<R::FixedRows<1>>(cellStatus));
            R::SubmitArrays(engine, stream, [](R::ExecutionContext c, auto b, auto s)
                            { K::MergeLVRBase(c, b, s, 0); }, R::ReadAs<R::FixedRows<2>>(dbase), R::WriteAs<R::TableRows>(db));
            R::SubmitArrays(engine, stream, [](R::ExecutionContext c, auto b, auto h, auto a)
                            { K::PublishLVRPenalty(c, b, h, a); }, R::ReadAs<R::TableRows>(db), R::WriteAs<R::FixedRows<2>>(dh), R::WriteAs<R::FixedRows<1>>(da));
            R::SubmitArrays(engine, stream, [gas](R::ExecutionContext c, auto u, auto s, auto status)
                            { K::FillLVRScalars(c, u, s, gas, status); }, R::ReadAs<R::TableRows>(dm), R::WriteAs<R::TableRows>(dscalar), R::WriteAs<R::FixedRows<1>>(scalarStatus));
            engine.Close();
            Good(faceStatus);
            Good(cellStatus);
            Good(scalarStatus);
            Equal(branches, db);
            Equal(base, dbase);
            Equal(history, dh);
            Equal(penalty, da);
            Equal(scalar, dscalar);
        }
}

TEST_CASE("CUDA O2 LVR target PP uses the retained pure-flow5 bound")
{
    const C::PerfectGas gas;
    const C::ThermalBounds bounds{1e-4, 1e-4, 1e-4};
    const auto state = gas.Conservative({1., .1, .2, 0, 1.});
    auto means = Table<real>(1, 5, [&](I, I j)
                             { return state[j]; });
    auto points = Table<real>(2, 5, [&](I iRow, I j)
                              { return j == 4 && iRow == 1 ? -2. : state[j]; });
    auto coefficients = Table<real>(1, 10, [](I, I j)
                                    { return .1 * (j + 1); });
    auto offsets = Table<I>(2, 1, [](I iRow, I)
                            { return 2 * iRow; });
    const auto device = R::Device::CUDA(0);
    auto dm = means.To(device), dp = points.To(device), dc = coefficients.To(device);
    auto dOffsets = offsets.To(device);
    R::Array<DNDS::NumericalStatus> status(R::RowLayout::Dense(1, 1), device);
    {
        const auto u = means.Read(), p = points.Read();
        const auto o = offsets.Read();
        auto c = coefficients.Write();
        REQUIRE(bool(C::CPU::LimitLVRTarget(u.View<R::UniformTableRows>(), p.View<R::UniformTableRows>(), c.View<R::UniformTableRows>(),
                                            o.View<R::FixedRows<1>>(), bounds, 1, 1e-3)));
    }
    R::EagerEngine engine;
    const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    R::SubmitArrays(engine, stream, [bounds](R::ExecutionContext c, auto u, auto p, auto coef, auto o, auto s)
                    { K::LimitLVRTarget(c, u, p, coef, o, bounds, 1, 1e-3, s); }, R::ReadAs<R::TableRows>(dm), R::ReadAs<R::TableRows>(dp), R::WriteAs<R::TableRows>(dc), R::ReadAs<R::FixedRows<1>>(dOffsets), R::WriteAs<R::FixedRows<1>>(status));
    engine.Close();
    Good(status);
    Equal(coefficients, dc);
    CHECK(coefficients.Read().View()(0, 0) < .1);
}
