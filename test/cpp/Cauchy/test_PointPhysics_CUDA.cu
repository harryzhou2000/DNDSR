#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Physics/RANS/Model.hpp"
#include "Physics/RANS/PointKernels.hpp"
#include "Cauchy/Physics/RiemannPreparation.hpp"
#include "Cauchy/Riemann/RiemannFluxMath.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include <cuda_runtime.h>

namespace R = DNDS::Runtime;
namespace P = DNDS::Physics::RANS;
namespace C = DNDS::Cauchy;
using I = DNDS::index;
using DNDS::real;

namespace
{
    template <class Row>
    DNDS_DEVICE_CALLABLE P::Point LoadPoint(const Row &row)
    {
        P::Point point;
        for (int j = 0; j < 7; ++j)
        {
            point.state[j] = row[j];
            for (int d = 0; d < 3; ++d)
                point.gradient(d, j) = row[7 + d * 7 + j];
        }
        point.molecularViscosity = row[28];
        point.wallDistance = row[29];
        point.cellLength = row[30];
        point.cp = row[31];
        point.gamma = row[32];
        return point;
    }

    template <class Input, class Output, class Status>
    __global__ void RANSPoints(P::PointParameters parameters, Input input, Output output, Status status)
    {
        for (I iRow = I(blockIdx.x) * blockDim.x + threadIdx.x; iRow < input.RowCount(); iRow += I(gridDim.x) * blockDim.x)
        {
            const auto point = LoadPoint(input[iRow]);
            const auto transport = P::TryTransport(parameters, point);
            const auto source = P::TrySource(parameters, point, 0);
            const auto diagonal = P::TrySource(parameters, point, 1);
            if (!transport || !source || !diagonal)
            {
                status(iRow, 0) = int(!transport ? transport.status.code : !source ? source.status.code
                                                                                   : diagonal.status.code);
                continue;
            }
            const auto diffusion = P::TryDiffusionFlux(parameters, point, Eigen::Vector3d(.6, .8, 0), transport.value.viscosity);
            status(iRow, 0) = int(diffusion.status.code);
            if (!diffusion)
                continue;
            output(iRow, 0) = transport.value.viscosity;
            output(iRow, 1) = transport.value.conductivity;
            output(iRow, 2) = transport.value.speciesDiffusivity;
            for (int j = 0; j < 2; ++j)
            {
                output(iRow, 3 + j) = diffusion.value[j];
                output(iRow, 5 + j) = source.value[j];
                output(iRow, 7 + j) = diagonal.value[j];
            }
        }
    }

    template <class Input, class Output>
    __global__ void RiemannPoints(C::PerfectGas gas, C::Riemann::Solver solver, Input input, Output output)
    {
        for (I iRow = I(blockIdx.x) * blockDim.x + threadIdx.x; iRow < input.RowCount(); iRow += I(gridDim.x) * blockDim.x)
        {
            C::Riemann::AcousticState states[4];
            for (int k = 0; k < 4; ++k)
            {
                for (int j = 0; j < 5; ++j)
                    states[k].flow.conservative[j] = input(iRow, 5 * k + j);
                states[k].flow.pressure = gas.Pressure(states[k].flow.conservative);
                states[k].soundSpeed = sqrt(gas.gamma * states[k].flow.pressure / states[k].flow.conservative[0]);
            }
            const C::Vector3 normal{.6, .8, 0}, grid{.03, -.02, .01};
            const auto prepared = C::PreparePerfectGasRiemann(gas, solver, states[0], states[1], states[2], states[3], normal, grid);
            const auto result = C::Riemann::Math::Evaluate(solver, prepared.Input(states[0], states[1], normal, grid), {.12, 1, .83, .1, .7});
            for (int j = 0; j < 5; ++j)
                output(iRow, j) = result.flux[j];
            output(iRow, 5) = result.acousticMinus;
            output(iRow, 6) = result.contact;
            output(iRow, 7) = result.acousticPlus;
        }
    }
}

TEST_CASE("Shared perfect-gas Riemann arithmetic agrees on CUDA for all schemes with distinct means and moving faces")
{
    constexpr I rows = 129;
    C::PerfectGas gas;
    R::Array<real> host(R::RowLayout::Dense(rows, 20));
    {
        auto access = host.Write();
        for (I iRow = 0; iRow < rows; ++iRow)
            for (int k = 0; k < 4; ++k)
            {
                const auto state = gas.Conservative({1 + .02 * k + .001 * iRow, -.7 + .02 * iRow + .03 * k,
                                                     .2 - .04 * k, .1, 1 + .01 * iRow + .1 * k});
                for (int j = 0; j < 5; ++j)
                    access.View()(iRow, 5 * k + j) = state[j];
            }
    }
    const auto device = R::Device::CUDA(0);
    const auto input = host.To(device);
    R::Array<real> output(R::RowLayout::Dense(rows, 8), device);
    for (int scheme = int(C::Riemann::Solver::Rusanov); scheme <= int(C::Riemann::Solver::AUSMPlusUp2); ++scheme)
    {
        CAPTURE(scheme);
        const auto solver = C::Riemann::Solver(scheme);
        R::EagerEngine engine;
        auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
        R::SubmitArrays(engine, stream, [gas, solver](auto context, auto x, auto y)
                        {
                            R::ValidateViewDevices(context.device, x, y);
                            R::CheckKernelArgumentBudget<decltype(gas), decltype(solver), decltype(x), decltype(y)>();
                            RiemannPoints<<<2, 128, 0, static_cast<cudaStream_t>(context.stream)>>>(gas, solver, x, y);
                            DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "Riemann point launch failed"); }, R::Read(input), R::Write(output));
        engine.Close();
        const auto downloaded = output.To(R::Device::Host());
        const auto actual = downloaded.Read(), values = host.Read();
        const C::Riemann::RiemannFlux flux(solver);
        for (I iRow = 0; iRow < rows; ++iRow)
        {
            C::Riemann::AcousticState states[4];
            for (int k = 0; k < 4; ++k)
            {
                for (int j = 0; j < 5; ++j)
                    states[k].flow.conservative[j] = values.View()(iRow, 5 * k + j);
                states[k].flow.pressure = gas.Pressure(states[k].flow.conservative);
                states[k].soundSpeed = std::sqrt(gas.gamma * states[k].flow.pressure / states[k].flow.conservative[0]);
            }
            const C::Vector3 normal{.6, .8, 0}, grid{.03, -.02, .01};
            const auto prepared = C::PreparePerfectGasRiemann(gas, solver, states[0], states[1], states[2], states[3], normal, grid);
            const auto request = prepared.Input(states[0], states[1], normal, grid);
            flux.Validate(request);
            const auto expected = flux.Evaluate(request, {.12, 1, .83, .1, .7});
            for (int j = 0; j < 5; ++j)
                CHECK(actual.View()(iRow, j) == doctest::Approx(expected.flux[j]).epsilon(3e-12).scale(1));
            CHECK(actual.View()(iRow, 5) == doctest::Approx(expected.acousticMinus).epsilon(3e-12).scale(1));
            CHECK(actual.View()(iRow, 6) == doctest::Approx(expected.contact).epsilon(3e-12).scale(1));
            CHECK(actual.View()(iRow, 7) == doctest::Approx(expected.acousticPlus).epsilon(3e-12).scale(1));
        }
    }
}

TEST_CASE("SA and k-omega CUDA point closures agree with canonical CPU source transport and diffusion")
{
    constexpr I rows = 129;
    R::Array<real> host(R::RowLayout::Dense(rows, 33));
    {
        auto access = host.Write();
        for (I iRow = 0; iRow < rows; ++iRow)
        {
            const real state[7]{1.3 + .001 * iRow, .4, -.2, .1, 3.2, .018, 2.7};
            for (int j = 0; j < 7; ++j)
            {
                access.View()(iRow, j) = state[j];
                for (int d = 0; d < 3; ++d)
                    access.View()(iRow, 7 + d * 7 + j) = .001 * ((j + 2 * d + iRow) % 17 - 8);
            }
            const real properties[5]{.0017, .09, .025, 3.5, 1.4};
            for (int j = 0; j < 5; ++j)
                access.View()(iRow, 28 + j) = properties[j];
        }
    }
    const auto device = R::Device::CUDA(0);
    const auto input = host.To(device);
    R::Array<real> output(R::RowLayout::Dense(rows, 9), device);
    R::Array<int> status(R::RowLayout::Dense(rows, 1), device);
    for (const char *name : {"SA", "KOmegaWilcox", "KOmegaSST"})
        for (int desMode = 0; desMode < 4; ++desMode)
        {
            CAPTURE(name);
            CAPTURE(desMode);
            P::ModelSettings settings;
            settings.model = name;
            settings.viscosityReference = .004;
            settings.desScale = desMode == 0 ? DNDS::veryLargeReal : .65;
            settings.desMode = desMode;
            settings.sst.implicitAdditionalSinks = true;
            const P::Model model(settings);
            const auto parameters = model.KernelParameters();
            R::EagerEngine engine;
            auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
            R::SubmitArrays(engine, stream, [parameters](auto context, auto x, auto y, auto s)
                            {
                                R::ValidateViewDevices(context.device, x, y, s);
                                R::CheckKernelArgumentBudget<decltype(parameters), decltype(x), decltype(y), decltype(s)>();
                                RANSPoints<<<2, 128, 0, static_cast<cudaStream_t>(context.stream)>>>(parameters, x, y, s);
                                DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "RANS point launch failed"); }, R::Read(input), R::Write(output), R::Write(status));
            engine.Close();
            const auto downloaded = output.To(R::Device::Host());
            const auto downloadedStatus = status.To(R::Device::Host());
            const auto actual = downloaded.Read(), values = host.Read();
            const auto codes = downloadedStatus.Read();
            for (I iRow = 0; iRow < rows; ++iRow)
            {
                REQUIRE(codes.View()(iRow, 0) == 0);
                const auto point = LoadPoint(values.View()[iRow]);
                const auto turbulent = model.Transport(point);
                const auto diffusion = model.DiffusionFlux(point, Eigen::Vector3d(.6, .8, 0), turbulent.viscosity);
                const auto source = model.Source(point), diagonal = model.ImplicitSourceDiagonal(point);
                const real expected[9]{turbulent.viscosity, turbulent.conductivity, turbulent.speciesDiffusivity,
                                       diffusion[0], diffusion[1], source[0], source[1], diagonal[0], diagonal[1]};
                for (int j = 0; j < 9; ++j)
                    CHECK(actual.View()(iRow, j) == doctest::Approx(expected[j]).epsilon(2e-11).scale(1e-8));
            }
        }
}
