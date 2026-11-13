#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Cauchy/Discretization/CUDA/FiniteVolumeAuxiliaryKernels.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumePointKernels.hpp"
#include "Cauchy/Config/SimulationConfig.hpp"
#include "Cauchy/Physics/EquationSystem.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
namespace K = DNDS::Cauchy::CUDA;
using I = DNDS::index;
using DNDS::real;

namespace
{
    template <class T>
    R::Array<T> DeviceRows(const std::vector<T> &values)
    {
        R::Array<T> host(R::RowLayout::Dense(I(values.size()), 1));
        {
            auto write = host.Write();
            for (I iRow = 0; iRow < I(values.size()); ++iRow)
                write.View()(iRow, 0) = values[iRow];
        }
        return host.To(R::Device::CUDA(0), R::ExecutionContext::CUDA(0));
    }
    R::Array<real> DeviceStates(const std::vector<C::FlowState> &values)
    {
        R::Array<real> host(R::RowLayout::Dense(I(values.size()), 5));
        {
            auto write = host.Write();
            for (I iRow = 0; iRow < I(values.size()); ++iRow)
                for (int j = 0; j < 5; ++j)
                    write.View()(iRow, j) = values[iRow][j];
        }
        return host.To(R::Device::CUDA(0), R::ExecutionContext::CUDA(0));
    }
}

TEST_CASE("CUDA auxiliary means match generic CPU thermodynamics and RANS transport with ghost gradients")
{
    const auto device = R::Device::CUDA(0);
    const auto context = R::ExecutionContext::CUDA(0);
    for (const std::string model : {"none", "SA", "KOmegaWilcox", "KOmegaSST"})
    {
        INFO("model=" << model);
        C::EquationSettings settings;
        settings.transport = "NewtonianFourier";
        settings.viscosity = .001;
        settings.viscosityModel = "sutherland";
        settings.sutherlandConstant = .3;
        settings.rans.model = model;
        settings.rans.viscosityReference = .01;
        const auto equation = C::EquationSystem::Resolve(settings);
        const int components = 5 + (equation.HasRANS() ? equation.RANS().ComponentNumber() : 0);
        const auto parameters = C::PerfectGasSpatialParameters::FromEquation(equation, 2);
        R::Array<C::PerfectGasSpatialParameters> hostParameters(R::RowLayout::Dense(1, 1));
        {
            auto write = hostParameters.Write();
            write.View()(0, 0) = parameters;
        }
        R::Array<real> own(R::RowLayout::Dense(3, components)), ghost(R::RowLayout::Dense(2, components));
        R::Array<real> ownGradient(R::RowLayout::Dense(3, components * 2)), ghostGradient(R::RowLayout::Dense(2, components * 2));
        R::Array<real> wall(R::RowLayout::Dense(5, 1)), expected(R::RowLayout::Dense(5, 3));
        {
            auto a = own.Write(), b = ghost.Write(), ga = ownGradient.Write(), gb = ghostGradient.Write();
            auto distance = wall.Write(), result = expected.Write();
            auto evaluator = equation.MakePointEvaluator();
            for (I iCell = 0; iCell < 5; ++iCell)
            {
                C::EquationFVDetail::PointScratch point(components);
                const auto flow = equation.Gas().Conservative({1 + .04 * iCell, .21, -.03 * iCell, .07, 1.1 + .01 * iCell});
                for (int j = 0; j < components; ++j)
                {
                    const auto value = j < 5 ? flow[j] : j == 5 ? .013
                                                                : .2;
                    point.values[j] = value;
                    const I row = iCell < 3 ? iCell : iCell - 3;
                    (iCell < 3 ? a.View() : b.View())(row, j) = value;
                    for (int d = 0; d < 2; ++d)
                    {
                        point.gradient(j, d) = .001 * (iCell + 1) * (j + 1) * (d == 0 ? 1 : -.3);
                        (iCell < 3 ? ga.View() : gb.View())(row, 2 * j + d) = point.gradient(j, d);
                    }
                }
                distance.View()(iCell, 0) = .03 + .01 * iCell;
                C::PointProperties properties;
                REQUIRE(bool(evaluator.TryEvaluate(flow, {}, 0, properties)));
                real bound = properties.diffusionBound;
                if (equation.HasRANS())
                {
                    const auto transport = equation.RANS().TryTransport(C::EquationFVDetail::RANSPoint(point, 0, 5, equation.RANS(), properties, distance.View()(iCell, 0), 0));
                    REQUIRE(bool(transport));
                    bound = std::max({bound, real(4. / 3.) * (properties.transport.viscosity + transport.value.viscosity) / flow[0],
                                      (properties.transport.conductivity + transport.value.conductivity) / (flow[0] * properties.thermo.cv)});
                }
                result.View()(iCell, 0) = properties.thermo.pressure;
                result.View()(iCell, 1) = properties.thermo.soundSpeed;
                result.View()(iCell, 2) = bound;
            }
        }
        auto p = hostParameters.To(device, context);
        auto u = own.To(device, context), ug = ghost.To(device, context);
        auto g = ownGradient.To(device, context), gg = ghostGradient.To(device, context), w = wall.To(device, context);
        R::Array<real> output(R::RowLayout::Dense(5, 3), device);
        R::Array<DNDS::NumericalStatus> status(R::RowLayout::Dense(5, 1), device);
        R::EagerEngine engine;
        const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
        // Both partitions are queued without a host join between them. The
        // same output lease is scheduled twice on one lane, with disjoint rows.
        for (const bool dependent : {false, true})
            R::SubmitArrays(engine, stream, [=](R::ExecutionContext execution, auto params, auto mean, auto halo, auto gradient, auto gradientHalo, auto walls, auto out, auto errors)
                            {
                K::AuxiliaryStateInput input{params, {mean, halo}, {gradient, gradientHalo}, walls, components, 2, true, true};
                K::PrepareAuxiliaryMeans(execution, input, out, errors, dependent ? 3 : 0, dependent ? 5 : 3); }, R::ReadAs<R::FixedRows<1>>(p), R::ReadAs<R::TableRows>(u), R::ReadAs<R::TableRows>(ug), R::ReadAs<R::TableRows>(g), R::ReadAs<R::TableRows>(gg), R::ReadAs<R::FixedRows<1>>(w), R::WriteAs<R::TableRows>(output), R::WriteAs<R::FixedRows<1>>(status));
        engine.Close();
        const auto result = output.To(R::Device::Host());
        const auto errors = status.To(R::Device::Host());
        const auto actual = result.Read(), reference = expected.Read();
        const auto error = errors.Read();
        for (I iCell = 0; iCell < 5; ++iCell)
        {
            REQUIRE(bool(error.View()(iCell, 0)));
            for (int j = 0; j < 3; ++j)
                CHECK(actual.View()(iCell, j) == doctest::Approx(reference.View()(iCell, j)).epsilon(2e-11).scale(1));
        }
    }
}

TEST_CASE("CUDA face auxiliaries distinguish rotated neighbor physical boundary and interface then reduce by cell")
{
    const auto device = R::Device::CUDA(0);
    const auto context = R::ExecutionContext::CUDA(0);
    C::PerfectGasSpatialParameters parameters;
    parameters.dimension = 2;
    parameters.Validate(5);
    const auto &gas = parameters.gas;
    const auto left = gas.Conservative({1., .3, -.1, .04, 1.});
    const auto neighbor = gas.Conservative({.8, .2, .4, -.03, .7});
    const auto incoming = gas.Conservative({1.2, -.15, .21, 0., 1.3});
    auto params = DeviceRows<C::PerfectGasSpatialParameters>({parameters});
    auto own = DeviceStates({left}), ghost = DeviceStates({neighbor}), packet = DeviceStates({incoming});
    auto entries = DeviceRows<I>({0, 1, 2}), faceIds = DeviceRows<I>({0, 1, 2}), quadratureOffsets = DeviceRows<I>({0, 1, 2, 3});
    auto selected = DeviceRows<I>({2, 0, 1}), cellOffsets = DeviceRows<I>({0, 3}), cellFaces = DeviceRows<I>({0, 1, 2}), selectedCells = DeviceRows<I>({0});
    auto area = DeviceRows<real>({1., .8, 1.2});
    auto kinds = DeviceRows<C::BoundaryKind>({C::BoundaryKind::Interior, C::BoundaryKind::FarField, C::BoundaryKind::Interface});
    R::Array<real> hostGeometry(R::RowLayout::Dense(2, 8));
    {
        auto write = hostGeometry.Write();
        for (I iCell = 0; iCell < 2; ++iCell)
            for (int j = 0; j < 8; ++j)
                write.View()(iCell, j) = j == 0 ? iCell + 1 : 0;
    }
    auto geometry = hostGeometry.To(device, context);
    std::vector<C::FiniteVolumeDetail::SampleRecord> records(3);
    for (I iFace = 0; iFace < 3; ++iFace)
    {
        auto &sample = records[iFace];
        sample.cell = 0;
        sample.other = iFace == 0 ? 1 : DNDS::UnInitIndex;
        sample.point = sample.face = sample.condition = iFace;
        sample.normal = {1, 0, 0};
        sample.ownPoint = {.2, .1 * iFace, 0};
        sample.rotation = {0, 1, 0, -1, 0, 0, 0, 0, 1};
        sample.changesFrame = iFace == 0;
        sample.boundaryPoint = 0;
    }
    auto samples = DeviceRows<C::FiniteVolumeDetail::SampleRecord>(records);
    R::Array<real> spectral(R::RowLayout::Dense(3, 2), device), wave(R::RowLayout::Dense(3, 1), device), pressure(R::RowLayout::Dense(3, 1), device);
    R::Array<DNDS::NumericalStatus> status(R::RowLayout::Dense(3, 1), device);
    C::RigidFrameParameters frame;
    frame.moving = true;
    frame.angularVelocity = {0, 0, .2};
    frame.centerVelocity = {.03, -.04, 0};
    {
        auto p = params.Read(context);
        auto u = own.Read(context), ug = ghost.Read(context), in = packet.Read(context), geo = geometry.Read(context), areas = area.Read(context);
        auto s = samples.Read(context);
        auto conditions = kinds.Read(context);
        auto e = entries.Read(context), f = faceIds.Read(context), q = quadratureOffsets.Read(context), selection = selected.Read(context);
        auto lambda = spectral.Write(context), jump = wave.Write(context), minimum = pressure.Write(context);
        auto errors = status.Write(context);
        K::AuxiliaryFaceRequest request;
        request.state.parameters = p.View<R::FixedRows<1>>();
        request.state.mean = {u.View<R::TableRows>(), ug.View<R::TableRows>()};
        request.state.dimension = 2;
        request.samples = s.View<R::FixedRows<1>>();
        request.entries = e.View<R::FixedRows<1>>();
        request.stencilFaces = f.View<R::FixedRows<1>>();
        request.quadratureOffsets = q.View<R::FixedRows<1>>();
        request.geometry = geo.View<R::FixedRows<8>>();
        request.area = areas.View<R::FixedRows<1>>();
        request.conditions = conditions.View<R::FixedRows<1>>();
        request.incoming = in.View<R::TableRows>();
        request.spectral = lambda.View<R::TableRows>();
        request.wave = jump.View<R::TableRows>();
        request.pressure = minimum.View<R::TableRows>();
        request.status = errors.View<R::FixedRows<1>>();
        request.frame = frame;
        request.needSpectral = request.needWave = request.needPressure = true;
        K::PrepareAuxiliaryFaces(context, request, selection.View<R::FixedRows<1>>());
        // This borrowed-view test deliberately owns its explicit completion
        // boundary. The launcher itself must not wait or release these leases.
        auto done = R::Completion::Record(context);
        lambda.Complete(done);
        jump.Complete(done);
        minimum.Complete(done);
        errors.Complete(done);
        done.Wait();
    }
    auto hSpectral = spectral.To(R::Device::Host()), hWave = wave.To(R::Device::Host()), hPressure = pressure.To(R::Device::Host());
    auto hStatus = status.To(R::Device::Host());
    const auto l = hSpectral.Read(), w = hWave.Read(), p = hPressure.Read();
    const auto errors = hStatus.Read();
    real sum = 0, maxJump = 0, minPressure = 1;
    for (I iFace = 0; iFace < 3; ++iFace)
    {
        REQUIRE(bool(errors.View()(iFace, 0)));
        auto right = iFace == 0 ? neighbor : iFace == 1 ? left
                                                        : incoming;
        if (iFace == 0)
        {
            right[1] = -neighbor[2];
            right[2] = neighbor[1];
        }
        const auto a = gas.Primitive(left), b = gas.Primitive(right);
        const real gridSpeed = .03 - .2 * (.1 * iFace);
        const real expectedLambda = std::max(std::abs(a.velocity[0] - gridSpeed) + a.soundSpeed,
                                             std::abs(b.velocity[0] - gridSpeed) + b.soundSpeed);
        const real expectedJump = std::abs(b.velocity[0] - a.velocity[0]) + std::abs(b.soundSpeed - a.soundSpeed);
        const real expectedPressure = std::min(a.pressure, b.pressure);
        CHECK(l.View()(iFace, 0) == doctest::Approx(expectedLambda));
        CHECK(l.View()(iFace, 1) == 0);
        CHECK(w.View()(iFace, 0) == doctest::Approx(expectedJump));
        CHECK(p.View()(iFace, 0) == doctest::Approx(expectedPressure));
        sum += std::array<real, 3>{1., .8, 1.2}[iFace] * expectedLambda;
        maxJump = std::max(maxJump, expectedJump);
        minPressure = std::min(minPressure, expectedPressure);
    }
    R::Array<real> cellLambda(R::RowLayout::Dense(1, 1), device), cellJump(R::RowLayout::Dense(1, 1), device), cellPressure(R::RowLayout::Dense(1, 1), device);
    {
        auto off = cellOffsets.Read(context), faces = cellFaces.Read(context), selection = selectedCells.Read(context);
        auto ar = area.Read(context), geo = geometry.Read(context), lambda = spectral.Read(context), jump = wave.Read(context), minimum = pressure.Read(context);
        auto ls = cellLambda.Write(context), jm = cellJump.Write(context), pm = cellPressure.Write(context);
        K::AuxiliaryCellRequest request{off.View<R::FixedRows<1>>(), faces.View<R::FixedRows<1>>(), ar.View<R::FixedRows<1>>(), geo.View<R::FixedRows<8>>(), lambda.View<R::TableRows>(), jump.View<R::TableRows>(), minimum.View<R::TableRows>(), {}, ls.View<R::TableRows>(), jm.View<R::TableRows>(), pm.View<R::TableRows>(), .2};
        K::ReduceAuxiliaryCells(context, request, selection.View<R::FixedRows<1>>());
        auto done = R::Completion::Record(context);
        ls.Complete(done);
        jm.Complete(done);
        pm.Complete(done);
        done.Wait();
    }
    const auto cl = cellLambda.To(R::Device::Host()), cj = cellJump.To(R::Device::Host()), cp = cellPressure.To(R::Device::Host());
    CHECK(cl.Read().View()(0, 0) == doctest::Approx(sum + .2));
    CHECK(cj.Read().View()(0, 0) == doctest::Approx(maxJump));
    CHECK(cp.Read().View()(0, 0) == doctest::Approx(minPressure));
}
