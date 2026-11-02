#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "CartesianFixture.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include <numeric>
#include <future>
#include <thread>
#include <chrono>

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
static DNDS::MPIInfo mpi;
namespace
{
    auto Arrays(const C::AuxiliaryValues &values)
    {
        return std::array<const R::Array<DNDS::real> *, 7>{
            values.meanThermodynamics ? &*values.meanThermodynamics : nullptr,
            values.faceSpectralRadius ? &*values.faceSpectralRadius : nullptr,
            values.faceWaveJump ? &*values.faceWaveJump : nullptr,
            values.facePressureMinimum ? &*values.facePressureMinimum : nullptr,
            values.cellSpectralSum ? &values.cellSpectralSum->Owned() : nullptr,
            values.cellWaveJump ? &values.cellWaveJump->Owned() : nullptr,
            values.cellPressureMinimum ? &values.cellPressureMinimum->Owned() : nullptr};
    }
    using Snapshot = std::array<std::vector<DNDS::real>, 7>;
    Snapshot Save(const C::AuxiliaryValues &values)
    {
        Snapshot saved;
        const auto arrays = Arrays(values);
        for (int k = 0; k < 7; ++k)
            if (arrays[k])
            {
                const auto read = arrays[k]->Read();
                const auto view = read.View<R::UniformTableRows>();
                for (DNDS::index iRow = 0; iRow < view.RowCount(); ++iRow)
                    for (DNDS::index j = 0; j < view.RowSize(iRow); ++j)
                        saved[k].push_back(view(iRow, j));
            }
        return saved;
    }
    void Compare(const Snapshot &expected, const C::AuxiliaryValues &actual)
    {
        const auto saved = Save(actual);
        const auto arrays = Arrays(actual);
        for (int k = 0; k < 7; ++k)
            if (arrays[k])
            {
                REQUIRE(saved[k].size() == expected[k].size());
                CHECK(saved[k] == expected[k]);
            }
    }
    R::RowLayout Layout(DNDS::index rows, DNDS::index width, int kind)
    {
        return kind == 0   ? R::RowLayout::Dense(rows, width, width + 3)
               : kind == 1 ? R::RowLayout::ColMajor(rows, width, {8, 1})
                           : R::RowLayout::TiledColMajor(rows, width, 8);
    }
    void Initialize(C::FiniteVolume &fv, DNDS::CFV::FVField &state, const C::EquationSystem &equation)
    {
        auto write = state.Owned().Write();
        const auto u = write.View<R::UniformTableRows>();
        for (DNDS::index iCell = 0; iCell < u.RowCount(); ++iCell)
        {
            const auto p = fv.Metrics().GetCellCent(iCell);
            const auto value = equation.Gas().Conservative({1.2 + .1 * std::sin(2 * DNDS::pi * p[0]),
                                                            .3 + .05 * std::sin(2 * DNDS::pi * p[1]),
                                                            -.2 + .03 * std::cos(2 * DNDS::pi * p[0]),
                                                            .04 * std::sin(2 * DNDS::pi * (p[0] + p[1])),
                                                            1.1 + .2 * std::cos(2 * DNDS::pi * p[1])});
            for (int j = 0; j < 5; ++j)
                u(iCell, j) = value[j];
        }
    }
    /// Native CPU-worker launcher with one controlled controller-side rejection.
    /// The first command is gated; its body is never run after an unsafe unwind
    /// in a red implementation. This tests joining, without inducing a use-after-free.
    class RejectAuxiliaryDependent final : public R::StreamLauncher
    {
        R::EagerEngine _worker{1};
        R::ExecutionStream _stream = _worker.CreateCPUStream();
        std::promise<void> _gate, _join;
        std::shared_future<void> _ready = _gate.get_future().share();
        std::atomic<bool> _released{false}, _joinRecorded{false};
        int _computes = 0;

    public:
        std::atomic<bool> rejected{false}, cancel{false}, coreCompleted{false};
        R::ExecutionContext Context() const override { return _stream.Context(); }
        void Submit(Work work, R::CommandKind kind, Retirement retire = {}) override
        {
            DNDS_check_throw_info(kind == R::CommandKind::Compute, "test launcher only accepts compute commands");
            const int command = ++_computes;
            if (command == 2)
            {
                rejected = true;
                throw std::runtime_error("controlled auxiliary dependent submission rejection");
            }
            if (command == 1)
                _worker.SubmitOperation(_stream, [this, work = std::move(work)](auto context)
                                        {
                    _ready.wait();
                    if (!cancel)
                    {
                        work(context);
                        coreCompleted = true;
                    } }, std::move(retire));
            else
                _worker.SubmitOperation(_stream, std::move(work), std::move(retire));
        }
        void WaitFor(R::StreamEvent event) override { _stream.WaitFor(std::move(event)); }
        R::StreamEvent Record() override
        {
            if (rejected && !_joinRecorded.exchange(true))
                _join.set_value();
            return _stream.Record();
        }
        void Drain() override { _worker.Synchronize(); }
        std::future<void> JoinRequested() { return _join.get_future(); }
        void Release()
        {
            if (!_released.exchange(true))
                _gate.set_value();
        }
        bool ObservedJoin() const { return _joinRecorded; }
    };
}
TEST_CASE("Auxiliary core completes during a real mean halo write and full publication waits for both phases")
{
    for (DNDS::index width : {DNDS::index(24), DNDS::index(2)})
    {
        const auto mesh = CauchyTest::PeriodicQuads(mpi, width);
        C::DomainSettings settings;
        settings.equation.transport = "NewtonianFourier";
        settings.equation.viscosity = .03;
        settings.numerics.cfv.maxOrder = 1;
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fv(mesh, settings, equation);
        const auto partition = DNDS::Geom::EntityWorkPartition::CellNodeCell(*mesh);
        DNDS::index count[2]{DNDS::index(partition.Core().size()), DNDS::index(partition.HaloDependent().size())};
        MPI_Allreduce(MPI_IN_PLACE, count, 2, MPI_INT64_T, MPI_SUM, mpi.comm);
        if (width == 24)
        {
            CHECK(count[0] > 0);
            if (mpi.size > 1)
                CHECK(count[1] > 0);
        }
        for (int layout = 0; layout < 3; ++layout)
        {
            auto state = fv.Metrics().BuildField(Layout(0, 5, layout));
            Initialize(fv, state, equation);
            auto halo = fv.Metrics().BindHalo(state);
            halo.Begin().Wait();
            auto reference = fv.BuildAuxiliaryValues({true, true, true, true, true, true, true});
            fv.PrepareAuxValues(state, reference, .25);
            const auto expected = Save(reference);
            double wave = std::accumulate(expected[2].begin(), expected[2].end(), 0.);
            MPI_Allreduce(MPI_IN_PLACE, &wave, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
            CHECK(wave > 0); // Nonconstant, nonzero signal fixture, including w.
            for (int selection = 0; selection < 6; ++selection)
                for (int workers : {0, 2})
                {
                    C::AuxiliaryRequest request;
                    request.meanThermodynamics = selection == 0 || selection == 4;
                    request.faceSpectralRadius = request.faceWaveJump = request.facePressureMinimum = selection < 2;
                    request.cellSpectralSum = request.cellWaveJump = request.cellPressureMinimum = selection == 0 || selection == 2 || selection == 5;
                    if (selection == 5)
                        request.faceSpectralRadius = request.facePressureMinimum = true;
                    auto output = fv.BuildAuxiliaryValues(request);
                    // Public outputs are caller-replaceable strided tables.
                    if (output.meanThermodynamics)
                        output.meanThermodynamics.emplace(Layout(fv.Metrics().NumCellProc(), 3, layout));
                    const DNDS::index faces = DNDS::index(fv.AuxiliaryFaces().size());
                    if (output.faceSpectralRadius)
                        output.faceSpectralRadius.emplace(Layout(faces, 2, layout));
                    if (output.faceWaveJump)
                        output.faceWaveJump.emplace(Layout(faces, 1, layout));
                    if (output.facePressureMinimum)
                        output.facePressureMinimum.emplace(Layout(faces, 1, layout));
                    for (auto *field : {&output.cellSpectralSum, &output.cellWaveJump, &output.cellPressureMinimum})
                        if (*field)
                            field->emplace(fv.Metrics().BuildField(Layout(0, 1, layout)));
                    const auto ownedVersion = state.Owned().Version();
                    const auto ghostVersion = state.Ghost().Version();
                    auto preparation = fv.BeginAuxValues(state, output, .25);
                    auto operation = halo.Begin();
                    R::EagerEngine engine(workers, std::make_shared<R::AccessTrackingContext>());
                    const auto stream = engine.CreateCPUStream();
                    std::atomic<bool> coreCompleted{false};
                    engine.Submit(stream, [&](auto)
                                  {
                        preparation.Apply(DNDS::CFV::ReconstructionRegion::Core);
                        coreCompleted = true; });
                    stream.Record().WaitHost();
                    CHECK(coreCompleted.load());
                    CHECK_THROWS(fv.ValidateAuxValues(state, output, .25));
                    CHECK_THROWS(preparation.Finish(output));
                    CHECK_THROWS(preparation.Apply(DNDS::CFV::ReconstructionRegion::Core));
                    if (state.Ghost().RowCount())
                    {
                        CHECK_THROWS(state.Ghost().Read());
                        CHECK_THROWS(preparation.Apply(DNDS::CFV::ReconstructionRegion::HaloDependent));
                    }
                    operation.Wait();
                    engine.Submit(stream, [&](auto)
                                  {
                        preparation.Apply(DNDS::CFV::ReconstructionRegion::HaloDependent);
                        preparation.Finish(output); });
                    stream.Record().WaitHost();
                    CHECK_NOTHROW(fv.ValidateAuxValues(state, output, .25));
                    CHECK_THROWS(preparation.Apply());
                    CHECK_THROWS(preparation.Finish(output));
                    CHECK(state.Owned().Version() == ownedVersion);
                    CHECK(state.Ghost().Version() >= ghostVersion);
                    Compare(expected, output);
                    auto released = state.Owned().Write();
                    released.CancelUnmodified();
                }
        }
    }
}
TEST_CASE("Auxiliary abandonment rebind and phase errors never publish a complete result")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 2);
    C::DomainSettings settings;
    settings.numerics.cfv.maxOrder = 1;
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::FiniteVolume fv(mesh, settings, equation);
    auto state = fv.Metrics().BuildField(5);
    Initialize(fv, state, equation);
    fv.Metrics().BindHalo(state).Begin().Wait();
    auto output = fv.BuildAuxiliaryValues({true, true, true, true, true, true, true});
    fv.PrepareAuxValues(state, output);
    const auto expected = Save(output);
    {
        auto abandoned = fv.BeginAuxValues(state, output);
        CHECK_THROWS(abandoned.Apply(DNDS::CFV::ReconstructionRegion::HaloDependent));
        abandoned.Apply(DNDS::CFV::ReconstructionRegion::Core);
        CHECK_THROWS(fv.ValidateAuxValues(state, output));
    }
    CHECK_THROWS(fv.ValidateAuxValues(state, output));
    auto preparation = fv.BeginAuxValues(state, output);
    preparation.Apply();
    const auto original = *output.meanThermodynamics;
    output.meanThermodynamics.emplace(R::RowLayout::Dense(fv.Metrics().NumCellProc(), 3));
    CHECK_THROWS(preparation.Finish(output));
    output.meanThermodynamics = original;
    preparation.Finish(output);
    CHECK_NOTHROW(fv.ValidateAuxValues(state, output));
    Compare(expected, output);
}
TEST_CASE("Unused private face scratch is safe when caller supplied outputs move to a larger face-only problem")
{
    C::DomainSettings settings;
    settings.numerics.cfv.maxOrder = 1;
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::FiniteVolume small(CauchyTest::PeriodicQuads(mpi, 2), settings, equation);
    // This selection allocates private face scratch. The next selection no
    // longer needs it and has many more faces; it must not write the old buffer.
    auto output = small.BuildAuxiliaryValues({false, false, false, false, true, true, true});
    output.cellSpectralSum.reset();
    output.cellWaveJump.reset();
    output.cellPressureMinimum.reset();
    C::FiniteVolume large(CauchyTest::PeriodicQuads(mpi, 24), settings, equation);
    auto state = large.Metrics().BuildField(5);
    Initialize(large, state, equation);
    large.Metrics().BindHalo(state).Begin().Wait();
    const auto faces = DNDS::index(large.AuxiliaryFaces().size());
    output.faceSpectralRadius.emplace(Layout(faces, 2, 1));
    output.faceWaveJump.emplace(Layout(faces, 1, 2));
    output.facePressureMinimum.emplace(Layout(faces, 1, 0));
    auto reference = large.BuildAuxiliaryValues({false, true, true, true, false, false, false});
    large.PrepareAuxValues(state, reference);
    large.PrepareAuxValues(state, output);
    CHECK_NOTHROW(large.ValidateAuxValues(state, output));
    Compare(Save(reference), output);
}
TEST_CASE("Auxiliary reductions count both incidences of a self-periodic face")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 1);
    C::DomainSettings settings;
    settings.numerics.cfv.maxOrder = 1;
    settings.equation.transport = "NewtonianFourier";
    settings.equation.viscosity = .03;
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::FiniteVolume fv(mesh, settings, equation);
    auto state = fv.Metrics().BuildField(5);
    const C::FlowState primitive{1.2, .3, -.2, .1, 1.1};
    fv.Initialize(state, {}, primitive);
    fv.Metrics().BindHalo(state).Begin().Wait();
    auto read = mesh->AcquireRead(DNDS::Geom::MeshReadRequirements::CellFlux());
    for (const auto iFace : fv.AuxiliaryFaces())
    {
        const auto parents = read.HostView().FaceCells(iFace);
        CHECK(parents[0] == parents[1]);
    }
    DNDS::index counts[]{fv.Metrics().NumCell(), DNDS::index(fv.AuxiliaryFaces().size())};
    MPI_Allreduce(MPI_IN_PLACE, counts, 2, MPI_INT64_T, MPI_SUM, mpi.comm);
    CHECK(counts[0] == 1);
    CHECK(counts[1] == 2);
    const double sound = std::sqrt(settings.equation.gamma * primitive[4] / primitive[0]);
    const double diffusion = settings.equation.viscosity / primitive[0] * std::max(4. / 3, settings.equation.gamma / settings.equation.prandtl);
    const double expected = 2 * (std::abs(primitive[1]) + std::abs(primitive[2]) + 2 * sound) + 8 * diffusion;
    for (bool faceOutput : {false, true})
    {
        auto output = fv.BuildAuxiliaryValues({false, faceOutput, false, false, true, false, false});
        fv.PrepareAuxValues(state, output);
        CHECK_NOTHROW(fv.ValidateAuxValues(state, output));
        const auto sums = output.cellSpectralSum->Owned().Read();
        for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
            CHECK(sums.View()(iCell, 0) == doctest::Approx(expected).epsilon(3e-13));
    }
}
TEST_CASE("Auxiliary transaction retains input output and geometry when original handles expire")
{
    C::AuxiliaryValues output;
    Snapshot expected;
    auto preparation = [&]
    {
        const auto mesh = CauchyTest::PeriodicQuads(mpi, 2);
        C::DomainSettings settings;
        settings.numerics.cfv.maxOrder = 1;
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fv(mesh, settings, equation);
        auto state = fv.Metrics().BuildField(5);
        Initialize(fv, state, equation);
        fv.Metrics().BindHalo(state).Begin().Wait();
        output = fv.BuildAuxiliaryValues({true, true, true, true, true, true, true});
        fv.PrepareAuxValues(state, output);
        expected = Save(output);
        return fv.BeginAuxValues(state, output);
    }();
    preparation.Apply();
    preparation.Finish(output);
    Compare(expected, output);
}
TEST_CASE("Auxiliary provenance includes DOF row identity when payload buffers are unchanged")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 2);
    C::DomainSettings settings;
    settings.numerics.cfv.maxOrder = 1;
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::FiniteVolume fv(mesh, settings, equation);
    auto state = fv.Metrics().BuildField(5);
    Initialize(fv, state, equation);
    fv.Metrics().BindHalo(state).Begin().Wait();
    const DNDS::Algebra::DofSpace other(state.Space().OwnedRows(), state.Space().GhostRows(), "different row identity");
    auto complete = fv.BuildAuxiliaryValues({false, false, false, false, true, false, false});
    fv.PrepareAuxValues(state, complete);
    const auto correct = *complete.cellSpectralSum;
    complete.cellSpectralSum.emplace(other, correct.Owned(), correct.Ghost());
    CHECK_THROWS(fv.ValidateAuxValues(state, complete));
    complete.cellSpectralSum = correct;
    CHECK_NOTHROW(fv.ValidateAuxValues(state, complete));
    auto pending = fv.BuildAuxiliaryValues({false, false, false, false, true, false, false});
    auto preparation = fv.BeginAuxValues(state, pending);
    preparation.Apply();
    const auto original = *pending.cellSpectralSum;
    pending.cellSpectralSum.emplace(other, original.Owned(), original.Ghost());
    CHECK_THROWS(preparation.Finish(pending));
}
TEST_CASE("Synchronous residual joins queued auxiliary work before a controller rejection unwinds its leases")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 2);
    C::DomainSettings settings;
    settings.numerics.reconstruction = "constant";
    settings.numerics.cfv.maxOrder = 1;
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::FiniteVolume fv(mesh, settings, equation);
    auto state = fv.Metrics().BuildField(5), reference = fv.Metrics().BuildField(5), result = fv.Metrics().BuildField(5);
    Initialize(fv, state, equation);
    auto work = fv.BuildWorkspace(state), referenceWork = fv.BuildWorkspace(state);
    R::EagerEngine baseline;
    fv.Evaluate(state, reference, *referenceWork, baseline, baseline.CreateCPUStream());
    auto launcher = std::make_shared<RejectAuxiliaryDependent>();
    R::EagerEngine engine(0, std::make_shared<R::AccessTrackingContext>());
    const auto stream = engine.CreateStream(launcher);
    auto requested = launcher->JoinRequested();
    std::atomic<bool> timeout{false};
    std::thread release([&]
                        {
        if (requested.wait_for(std::chrono::seconds(10)) != std::future_status::ready)
        {
            timeout = true;
            launcher->cancel = true;
        }
        launcher->Release(); });
    CHECK_THROWS(fv.Evaluate(state, result, *work, engine, stream));
    // A missing join fails safely: skip the delayed body before permitting the
    // worker to resume, instead of intentionally dereferencing expired leases.
    if (!launcher->ObservedJoin())
    {
        launcher->cancel = true;
        launcher->Release();
    }
    release.join();
    CHECK_FALSE(timeout);
    CHECK(launcher->rejected);
    CHECK(launcher->ObservedJoin());
    CHECK(launcher->coreCompleted);
    CHECK_THROWS(fv.PreparedAuxValues(*work));
    {
        auto released = state.Owned().Write();
        released.CancelUnmodified();
    }
    CHECK_THROWS_WITH(fv.Evaluate(state, result, *work, engine, stream), doctest::Contains("failed allocation must be replaced"));
    // Runtime poisons partial write allocations during exception unwinding.
    // Keep the accepted input and engine, but replace failed numerical scratch.
    work = fv.BuildWorkspace(state);
    REQUIRE_NOTHROW(fv.Evaluate(state, result, *work, engine, stream));
    const auto expected = reference.Owned().Read(), actual = result.Owned().Read();
    for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
        for (int j = 0; j < 5; ++j)
            CHECK(actual.View()(iCell, j) == expected.View()(iCell, j));
}
int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    mpi.setWorld();
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const int result = context.run();
    MPI_Finalize();
    return result;
}
