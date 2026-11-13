#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Cauchy/Discretization/CUDA/FiniteVolumePositivityKernels.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumePositivityKernels.hpp"
#include "Cauchy/Config/SimulationConfig.hpp"
#include <cuda_runtime.h>

namespace C = DNDS::Cauchy;
namespace K = DNDS::Cauchy::CUDA;
namespace R = DNDS::Runtime;
using I = DNDS::index;
using DNDS::real;

namespace
{
    template <class T, class F>
    R::Array<T> Table(I rows, I width, F f)
    {
        R::Array<T> array(R::RowLayout::Dense(rows, width));
        auto write = array.Write();
        for (I iRow = 0; iRow < rows; ++iRow)
            for (I j = 0; j < width; ++j)
                write.View()(iRow, j) = f(iRow, j);
        return array;
    }
    template <class T>
    auto Components(R::ArrayView<T, R::AccessBoundsChecks, R::UniformTableRows> view)
    {
        std::vector<DNDS::Algebra::DofComponentColumn<T>> columns;
        for (I j = 0; j < view.Width(); ++j)
            columns.push_back({view, {}, j});
        return DNDS::Algebra::DofComponentLayout<T>(std::move(columns), view.RowCount(), 0);
    }
    R::Array<real> Sample(const R::Array<real> &mean, const R::Array<real> &coefs)
    {
        const auto u = mean.Read(), r = coefs.Read();
        const I width = mean.Layout().Width(), modes = coefs.Layout().Width() / width;
        return Table<real>(2 * mean.RowCount(), width, [&](I q, I j)
                           {
            real value = u.View()(q / 2, j);
            for (I m = 0; m < modes; ++m)
                value += (q % 2 ? -1 : 1) * r.View()(q / 2, j * modes + m) / (m + 1);
            return value; });
    }
    template <class T>
    void Compare(const R::Array<T> &device, const R::Array<T> &cpu)
    {
        const auto downloaded = device.To(R::Device::Host());
        const auto a = downloaded.Read(), b = cpu.Read();
        for (I i = 0; i < cpu.RowCount(); ++i)
            for (I j = 0; j < cpu.Layout().Width(); ++j)
                CHECK(a.View()(i, j) == doctest::Approx(real(b.View()(i, j))).epsilon(2e-12).scale(1));
    }
}

TEST_CASE("CUDA physical-dt forecast ignores RANS bounds and reports invalid means")
{
    C::EquationSettings settings;
    settings.rans.model = "KOmegaSST";
    const auto equation = C::EquationSystem::Resolve(settings);
    C::ThermalPolicy thermal(equation, 1e-6, 1e-6, 0, 1e-6);
    auto means = Table<real>(3, 7, [&](I, I j)
                             {
        const auto flow = equation.Gas().Conservative({1.2, .2, .1, -.1, 1.1});
        return j < 5 ? flow[j] : .02; });
    auto rhs = Table<real>(3, 7, [](I i, I j)
                           {
        if (j >= 5) return real(-1e20);
        return i == 1 && j == 4 ? real(-100) : i == 2 && j == 0 ? real(-10) : real(0); });
    auto geometry = Table<real>(3, 8, [](I i, I j)
                                { return j == 0 ? real(i + 1) : real(0); });
    std::pair<I, real> expected;
    {
        const auto u = means.Read(), r = rhs.Read(), g = geometry.Read();
        const auto meanLayout = Components(u.View<R::UniformTableRows>()), rhsLayout = Components(r.View<R::UniformTableRows>());
        DNDS::CFV::FiniteVolumeHostView metrics;
        metrics.cells = g.View<R::FixedRows<8>, false>();
        const auto result = C::CPU::LimitPhysicalStepRows(meanLayout.View(), rhsLayout.View(), metrics, thermal.View(), .1, .5, 1, 1);
        REQUIRE(bool(result));
        expected = result.value;
    }
    const auto device = R::Device::CUDA(0), host = R::Device::Host();
    const auto context = R::ExecutionContext::CUDA(0);
    auto du = means.To(device), dr = rhs.To(device), dg = geometry.To(device);
    R::Array<real> alpha(R::RowLayout::Dense(3, 1), device);
    R::Array<DNDS::NumericalStatus> status(R::RowLayout::Dense(3, 1), device);
    const auto launch = [&]
    {
        const auto u = du.Read(context), r = dr.Read(context), g = dg.Read(context);
        auto a = alpha.Write(context);
        auto s = status.Write(context);
        K::LimitPhysicalStepRows(context, u.View<R::TableRows>(), r.View<R::TableRows>(), g.View<R::FixedRows<8>, false>(), thermal.View(), .1, .5, 1, 1, a.View<R::FixedRows<1>>(), s.View<R::FixedRows<1>>());
        REQUIRE(cudaStreamSynchronize(static_cast<cudaStream_t>(context.stream)) == cudaSuccess);
    };
    launch();
    {
        const auto aHost = alpha.To(host);
        const auto sHost = status.To(host);
        const auto a = aHost.Read();
        const auto s = sHost.Read();
        I count = 0;
        real minimum = 1;
        for (I i = 0; i < 3; ++i)
        {
            REQUIRE(bool(s.View()(i, 0)));
            if (a.View()(i, 0) < 1)
            {
                ++count;
                minimum = std::min(minimum, a.View()(i, 0));
            }
        }
        CHECK(a.View()(0, 0) == 1); // Only scalar components have a wild forecast.
        CHECK(count == expected.first);
        CHECK(minimum == doctest::Approx(expected.second).epsilon(2e-12));
    }
    {
        auto u = means.Write();
        u.View()(0, 0) = -1;
    }
    du.CopyFrom(means, context).Wait();
    launch();
    const auto errors = status.To(host);
    const auto read = errors.Read();
    CHECK(read.View()(0, 0).code == DNDS::NumericalError::InvalidState);
    CHECK(read.View()(0, 0).row == 0);
}

TEST_CASE("CUDA polynomial positivity matches CPU flow compression and independent RANS repair")
{
    const auto device = R::Device::CUDA(0), host = R::Device::Host();
    const auto context = R::ExecutionContext::CUDA(0);
    for (const std::string model : {"none", "SA", "KOmegaSST"})
        for (const auto policy : {C::ReconstructionClippedReferencePolicy::FallbackToBase, C::ReconstructionClippedReferencePolicy::Backtracking})
        {
            INFO("model=" << model << ", recovery=" << int(policy));
            C::EquationSettings settings;
            settings.rans.model = model;
            const auto equation = C::EquationSystem::Resolve(settings);
            const int width = 5 + (equation.HasRANS() ? equation.RANS().ComponentNumber() : 0), modes = 9;
            C::ThermalPolicy thermal(equation, 1e-6, 1e-6, 0, 1e-6);
            auto means = Table<real>(3, width, [&](I, I j)
                                     {
                const auto flow = equation.Gas().Conservative({1.2, .2, .1, -.1, 1.1});
                return j < 5 ? flow[j] : j == 5 ? .02 : .3; });
            auto coefficients = Table<real>(3, width * modes, [&](I iCell, I j)
                                            {
                const I component = j / modes, mode = j % modes;
                if (mode != 0) return real(0);
                if (iCell == 1 && component == 0) return real(-2);
                if (iCell == 2 && component == 4) return real(-10);
                if (iCell == 2 && component == 5) return real(-.1);
                return real(.0001); });
            const auto initial = coefficients.To(host);
            auto reference = coefficients.To(host); // To(same device) makes a value copy.
            auto higher = Sample(means, coefficients);
            auto offsets = Table<I>(4, 1, [](I i, I)
                                    { return 2 * i; });
            R::Array<I> degrees(R::RowLayout::Dense(3, 1));
            R::Array<real> beta(R::RowLayout::Dense(3, 1)), scale(beta.Layout());
            R::Array<uint8_t> recover(beta.Layout());
            auto dmeans = means.To(device), dcoefs = coefficients.To(device), dref = reference.To(device), dhigh = higher.To(device);
            auto doff = offsets.To(device);
            R::Array<I> ddegrees(degrees.Layout(), device);
            R::Array<real> dbeta(beta.Layout(), device), dscale(beta.Layout(), device);
            R::Array<uint8_t> dgood(beta.Layout(), device), drecover(beta.Layout(), device);
            R::Array<DNDS::NumericalStatus> dstatus(beta.Layout(), device);
            const auto meanRead = means.Read(), higherRead = higher.Read();
            const auto offsetRead = offsets.Read();
            const auto meanLayout = Components(meanRead.View<R::UniformTableRows>()), highLayout = Components(higherRead.View<R::UniformTableRows>());
            C::CPU::PolynomialPositivityRequest cpu;
            cpu.means = meanLayout.View();
            cpu.offsets = offsetRead.View<R::FixedRows<1>>();
            cpu.bounds = thermal.View();
            cpu.rans = C::ResolveRANSReconstructionPolicy(equation);
            cpu.ransFirst = width > 5 ? 5 : -1;
            cpu.modes = modes;
            cpu.maxOrder = 3;
            cpu.betaPower = 1;
            cpu.betaCutoff = 1e-8;
            cpu.clippedPolicy = policy;
            const auto um = dmeans.Read(context);
            const auto of = doff.Read(context);
            std::optional<R::Array<DNDS::NumericalStatus>::WriteLease> status(dstatus.Write(context));
            K::PolynomialPositivityRequest gpu;
            gpu.means = um.View<R::TableRows>();
            gpu.offsets = of.View<R::FixedRows<1>>();
            gpu.status = status->View<R::FixedRows<1>>();
            gpu.policy.bounds = thermal.View();
            gpu.policy.rans = cpu.rans;
            gpu.policy.components = width;
            gpu.policy.modes = modes;
            gpu.policy.maxOrder = 3;
            gpu.policy.betaPower = 1;
            gpu.policy.betaCutoff = cpu.betaCutoff;
            gpu.policy.clippedPolicy = policy;
            const auto checkStatus = [&]
            {
                REQUIRE(cudaStreamSynchronize(static_cast<cudaStream_t>(context.stream)) == cudaSuccess);
                status.reset();
                {
                    const auto downloaded = dstatus.To(host);
                    const auto read = downloaded.Read();
                    for (I i = 0; i < 3; ++i)
                        REQUIRE(bool(read.View()(i, 0)));
                }
                status.emplace(dstatus.Write(context));
                gpu.status = status->View<R::FixedRows<1>>();
            };
            REQUIRE(bool(C::CPU::ValidateReconstructionMeans(cpu)));
            K::ValidateReconstructionMeans(context, gpu);
            checkStatus();
            {
                auto d = degrees.Write();
                const auto marked = C::CPU::MarkAdmissibleDegrees(cpu, highLayout.View(), d.View<R::FixedRows<1>>());
                REQUIRE(bool(marked));
                CHECK_FALSE(marked.value);
                const auto h = dhigh.Read(context);
                auto dd = ddegrees.Write(context);
                auto good = dgood.Write(context);
                K::MarkAdmissibleDegrees(context, gpu, h.View<R::TableRows>(), dd.View<R::FixedRows<1>>(), good.View<R::FixedRows<1>>());
                checkStatus();
            }
            Compare(ddegrees, degrees);
            {
                const auto d = degrees.Read();
                auto ref = reference.Write();
                const auto layout = Components(ref.View<R::UniformTableRows>());
                C::CPU::TruncateReferenceModes(cpu, d.View<R::FixedRows<1>>(), layout.View(), 0);
                const auto dd = ddegrees.Read(context);
                auto dr = dref.Write(context);
                K::TruncateReferenceModes(context, gpu, dd.View<R::FixedRows<1>>(), dr.View<R::TableRows>(), 0);
                checkStatus();
            }
            auto base = Sample(means, reference), dbase = base.To(device);
            const auto baseRead = base.Read();
            const auto baseLayout = Components(baseRead.View<R::UniformTableRows>());
            {
                const auto ref = reference.Read();
                auto d = degrees.Write();
                auto c = coefficients.Write();
                const auto refLayout = Components(ref.View<R::UniformTableRows>());
                const auto out = Components(c.View<R::UniformTableRows>());
                C::CPU::SelectReferenceDegree(cpu, baseLayout.View(), refLayout.View(), d.View<R::FixedRows<1>>(), out.View(), 0);
                const auto b = dbase.Read(context), r = dref.Read(context);
                auto dd = ddegrees.Write(context);
                auto dc = dcoefs.Write(context);
                K::SelectReferenceDegree(context, gpu, b.View<R::TableRows>(), r.View<R::TableRows>(), dd.View<R::FixedRows<1>>(), dc.View<R::TableRows>(), 0);
                checkStatus();
            }
            {
                const auto ref = reference.Read();
                const auto d = degrees.Read();
                auto c = coefficients.Write(), b = beta.Write();
                const auto refLayout = Components(ref.View<R::UniformTableRows>());
                const auto out = Components(c.View<R::UniformTableRows>());
                REQUIRE(bool(C::CPU::CompressPolynomial(cpu, highLayout.View(), baseLayout.View(), refLayout.View(), d.View<R::FixedRows<1>>(), out.View(), b.View<R::FixedRows<1>>())));
                const auto h = dhigh.Read(context), br = dbase.Read(context), rr = dref.Read(context);
                const auto dd = ddegrees.Read(context);
                auto dc = dcoefs.Write(context), db = dbeta.Write(context);
                K::CompressPolynomial(context, gpu, h.View<R::TableRows>(), br.View<R::TableRows>(), rr.View<R::TableRows>(), true, true, dd.View<R::FixedRows<1>>(), dc.View<R::TableRows>(), db.View<R::FixedRows<1>>());
                checkStatus();
            }
            Compare(dcoefs, coefficients);
            Compare(dbeta, beta);
            // Re-evaluate actual stored coefficients, rather than accepting
            // the computed beta as a proof of admissibility.
            auto finalSamples = Sample(means, coefficients), dfinal = finalSamples.To(device);
            {
                const auto s = finalSamples.Read();
                const auto layout = Components(s.View<R::UniformTableRows>());
                REQUIRE(bool(C::CPU::VerifyReconstructedPoints(cpu, layout.View())));
                const auto ds = dfinal.Read(context);
                K::VerifyReconstructedPoints(context, gpu, ds.View<R::TableRows>());
                checkStatus();
            }
            // Explicit recovery input: the uncompressed samples correspond to
            // an initially unit beta. Compare one controller recovery pass for
            // both policies; backtracking need not finish in a single pass.
            {
                auto b = beta.Write();
                for (I i = 0; i < 3; ++i)
                    b.View()(i, 0) = 1;
                // Upload outside any kernel; this test owns the sync boundary.
            }
            dbeta.CopyFrom(beta, context).Wait();
            coefficients.CopyFrom(initial).Wait();
            dcoefs.CopyFrom(initial, context).Wait();
            {
                auto b = beta.Write(), s = scale.Write();
                auto rec = recover.Write();
                REQUIRE(bool(C::CPU::MarkPolynomialRecovery(cpu, highLayout.View(), b.View<R::FixedRows<1>>(), rec.View<R::FixedRows<1>>(), s.View<R::FixedRows<1>>())));
                const auto h = dhigh.Read(context);
                auto db = dbeta.Write(context), ds = dscale.Write(context);
                auto dr = drecover.Write(context);
                K::MarkPolynomialRecovery(context, gpu, h.View<R::TableRows>(), db.View<R::FixedRows<1>>(), dr.View<R::FixedRows<1>>(), ds.View<R::FixedRows<1>>());
                checkStatus();
            }
            Compare(drecover, recover);
            Compare(dbeta, beta);
            // Only flagged rows read recoveryScale; inactive entries remain
            // unspecified and must not be compared or reduced as valid data.
            {
                const auto ref = reference.Read(), s = scale.Read();
                const auto rec = recover.Read();
                auto c = coefficients.Write();
                const auto refLayout = Components(ref.View<R::UniformTableRows>());
                const auto out = Components(c.View<R::UniformTableRows>());
                C::CPU::RecoverPolynomial(cpu, refLayout.View(), out.View(), rec.View<R::FixedRows<1>>(), s.View<R::FixedRows<1>>());
                const auto rr = dref.Read(context), ds = dscale.Read(context);
                const auto dr = drecover.Read(context);
                auto dc = dcoefs.Write(context);
                K::RecoverPolynomial(context, gpu, rr.View<R::TableRows>(), true, dc.View<R::TableRows>(), dr.View<R::FixedRows<1>>(), ds.View<R::FixedRows<1>>());
                checkStatus();
            }
            Compare(dcoefs, coefficients);
        }
}
