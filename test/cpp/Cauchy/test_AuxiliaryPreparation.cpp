#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "CartesianFixture.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include <algorithm>
#include <numeric>
#include <future>
#include <thread>
#include <chrono>

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
static DNDS::MPIInfo mpi;
namespace
{
    C::EquationDofs Bind(const C::FiniteVolume &fv, const DNDS::CFV::FVField &field)
    {
        return C::EquationDofs::Coupled(fv.Equation().Schema(), field);
    }
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
    /// The core command runs on a real worker; the following dependent command
    /// is rejected before acceptance. Correct unwinding must leave no pending
    /// accesses from the completed prefix.
    class RejectAuxiliaryDependent final : public R::StreamLauncher
    {
        R::EagerEngine _worker{1};
        R::ExecutionStream _stream = _worker.CreateCPUStream();
        std::atomic<bool> _joinRecorded{false};
        int _computes = 0;

    public:
        std::atomic<bool> rejected{false}, coreCompleted{false};
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
            _worker.SubmitOperation(_stream, [this, work = std::move(work)](auto context)
                                    {
                work(context);
                coreCompleted = true; }, std::move(retire));
        }
        void WaitFor(R::StreamEvent event) override { _stream.WaitFor(std::move(event)); }
        R::StreamEvent Record() override
        {
            if (rejected)
                _joinRecorded = true;
            return _stream.Record();
        }
        void Drain() override { _worker.Synchronize(); }
        bool ObservedJoin() const { return _joinRecorded; }
    };
    /// Record the real FV submission protocol while executing through the
    /// ordinary CPU provider. This checks launcher independence and event
    /// placement, not GPU arithmetic or a captured CUDA graph.
    class RecordingAuxiliaryLauncher final : public R::StreamLauncher
    {
        R::EagerEngine _engine;
        R::ExecutionStream _stream;

    public:
        std::vector<std::string> trace;
        explicit RecordingAuxiliaryLauncher(int workers)
            : _engine(workers), _stream(_engine.CreateCPUStream()) {}
        R::ExecutionContext Context() const override { return _stream.Context(); }
        void Submit(Work work, R::CommandKind kind, Retirement retire = {}) override
        {
            DNDS_check_throw_info(kind == R::CommandKind::Compute,
                                  "recording FV launcher expects compute work");
            trace.emplace_back("submit");
            _engine.SubmitOperation(_stream, std::move(work), std::move(retire));
        }
        void WaitFor(R::StreamEvent event) override
        {
            trace.emplace_back("wait");
            _stream.WaitFor(std::move(event));
        }
        R::StreamEvent Record() override
        {
            trace.emplace_back("record");
            return _stream.Record();
        }
        void Drain() override
        {
            trace.emplace_back("drain");
            _engine.Synchronize();
        }
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
            fv.PrepareAuxValues(Bind(fv, state), reference, .25);
            const auto expected = Save(reference);
            // The logical-state producer preserves the established flow5
            // auxiliary values exactly, without repacking the physical table.
            auto semantic = C::EquationDofs::Coupled(equation.Schema(), state);
            auto logical = fv.BuildAuxiliaryValues({true, true, true, true, true, true, true});
            auto logicalExchange = halo.Begin();
            auto logicalPreparation = fv.BeginAuxValues(semantic, logical, .25);
            logicalPreparation.Apply(DNDS::CFV::ReconstructionRegion::Core);
            CHECK_THROWS(logicalPreparation.Apply(DNDS::CFV::ReconstructionRegion::HaloDependent));
            logicalExchange.Wait();
            logicalPreparation.Apply(DNDS::CFV::ReconstructionRegion::HaloDependent);
            logicalPreparation.Finish(logical);
            fv.ValidateAuxValues(semantic, logical, .25);
            Compare(expected, logical);
            auto reduced = fv.BuildAuxiliaryValues({false, false, false, false, true, true, true});
            fv.PrepareAuxValues(semantic, reduced, .25);
            fv.ValidateAuxValues(semantic, reduced, .25);
            Compare(expected, reduced);
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
                    auto preparation = fv.BeginAuxValues(Bind(fv, state), output, .25);
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
                    CHECK_THROWS(fv.ValidateAuxValues(Bind(fv, state), output, .25));
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
                    CHECK_NOTHROW(fv.ValidateAuxValues(Bind(fv, state), output, .25));
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
TEST_CASE("Queued CPU auxiliary accepts pending means and feeds consumers before publication")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 6);
    C::DomainSettings settings;
    settings.equation.transport = "NewtonianFourier";
    settings.equation.viscosity = .03;
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::FiniteVolume fv(mesh, settings, equation);
    auto state = fv.Metrics().BuildField(5);
    Initialize(fv, state, equation);
    auto logical = Bind(fv, state);
    auto halo = fv.Metrics().BindHalo(state);
    halo.Begin().Wait();
    struct Gate final : R::StreamEvent::State
    {
        std::promise<void> promise;
        std::shared_future<void> future = promise.get_future().share();
        bool Ready() const override { return future.wait_for(std::chrono::seconds(0)) == std::future_status::ready; }
        void Wait() const override { future.wait(); }
        void Signal()
        {
            promise.set_value();
            NotifyCompletion();
        }
    };
    for (const bool derived : {false, true})
        for (const int workers : {0, 2})
        {
            C::AuxiliaryRequest request{true, true, true, true, true, true, true};
            if (derived)
                request.derived = {C::DerivedQuantity::Temperature};
            auto reference = fv.BuildAuxiliaryValues(request), output = fv.BuildAuxiliaryValues(request);
            fv.PrepareAuxValues(logical, reference, .25);
            const auto expected = Save(reference);
            R::EagerEngine engine(workers, std::make_shared<R::AccessTrackingContext>());
            const auto compute = engine.CreateCPUStream(), communication = engine.CreateCPUStream();
            const auto gate = std::make_shared<Gate>();
            struct Unblock
            {
                std::shared_ptr<Gate> gate;
                ~Unblock()
                {
                    if (!gate->Ready())
                        gate->Signal();
                }
            } unblock{gate};
            compute.WaitFor(R::StreamEvent(gate));
            R::SubmitArrays(engine, compute, [](R::ExecutionContext, auto values)
                            {
                for (DNDS::index iRow = 0; iRow < values.RowCount(); ++iRow)
                    for (DNDS::index j = 0; j < values.Width(); ++j) values(iRow, j) += 0.; }, R::ReadWriteAs<R::TableRows>(state.Owned()));
            // Begin must accept the scheduled producer. It retains handles,
            // not an immediate source read or output writer.
            auto preparation = fv.BeginAuxValues(logical, output, .25);
            CHECK_THROWS(state.Owned().Read());
            gate->Signal();
            auto transfer = halo.SubmitAfter(engine, communication, compute.Record());
            preparation.Apply(engine, compute, DNDS::CFV::ReconstructionRegion::Core);
            transfer.EnqueueWait(compute);
            preparation.Apply(engine, compute, DNDS::CFV::ReconstructionRegion::HaloDependent);
            R::Array<double> observed(output.cellSpectralSum->Owned().Layout());
            R::SubmitArrays(engine, compute, [](R::ExecutionContext, auto input, auto output)
                            {
                for (DNDS::index iRow = 0; iRow < output.RowCount(); ++iRow) output(iRow, 0) = input(iRow, 0); }, R::ReadAs<R::TableRows>(output.cellSpectralSum->Owned()), R::WriteAs<R::TableRows>(observed));
            engine.Synchronize();
            preparation.Finish(output);
            fv.ValidateAuxValues(logical, output, .25);
            Compare(expected, output);
            const auto sums = observed.Read();
            for (DNDS::index iRow = 0; iRow < observed.RowCount(); ++iRow)
                CHECK(sums.View()(iRow, 0) == expected[4][iRow]);
            if (derived)
            {
                CHECK(output.derivedOwned->Current(logical, equation));
                CHECK(output.derivedGhost->Current(logical, equation));
            }
        }
}

TEST_CASE("High-order logical auxiliary maxima match the full pointwise flow path")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 6);
    for (bool moving : {false, true})
    {
        C::DomainSettings settings;
        settings.numerics.reconstruction = "variational";
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.cfv.intOrder = 5;
        settings.equation.transport = "NewtonianFourier";
        settings.equation.viscosity = .03;
        if (moving)
            settings.motion.angularVelocity = {0, 0, .4};
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fv(mesh, settings, equation);
        const auto geometry = fv.Metrics().AcquireRead();
        const auto metric = geometry.HostView();
        if (fv.Metrics().NumFaceProc())
            REQUIRE(metric.FaceQuadratureEnd(0) - metric.FaceQuadratureBegin(0) > 1);
        auto state = fv.Metrics().BuildField(5);
        Initialize(fv, state, equation);
        fv.BindHalo(state).Begin().Wait();
        auto reference = fv.BuildAuxiliaryValues({true, true, true, true, true, true, true});
        auto observed = fv.BuildAuxiliaryValues({true, true, true, true, true, true, true});
        fv.PrepareAuxValues(Bind(fv, state), reference, .25);
        const auto logical = C::EquationDofs::Coupled(equation.Schema(), state);
        fv.PrepareAuxValues(logical, observed, .25);
        Compare(Save(reference), observed);
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
    fv.PrepareAuxValues(Bind(fv, state), output);
    const auto expected = Save(output);
    {
        auto abandoned = fv.BeginAuxValues(Bind(fv, state), output);
        CHECK_THROWS(abandoned.Apply(DNDS::CFV::ReconstructionRegion::HaloDependent));
        abandoned.Apply(DNDS::CFV::ReconstructionRegion::Core);
        CHECK_THROWS(fv.ValidateAuxValues(Bind(fv, state), output));
    }
    CHECK_THROWS(fv.ValidateAuxValues(Bind(fv, state), output));
    // Abandoning a transaction poisons its partially reserved outputs. A
    // retry must use replacement storage rather than silently reusing them.
    output = fv.BuildAuxiliaryValues({true, true, true, true, true, true, true});
    auto preparation = fv.BeginAuxValues(Bind(fv, state), output);
    preparation.Apply();
    const auto original = *output.meanThermodynamics;
    output.meanThermodynamics.emplace(R::RowLayout::Dense(fv.Metrics().NumCellProc(), 3));
    CHECK_THROWS(preparation.Finish(output));
    output.meanThermodynamics = original;
    preparation.Finish(output);
    CHECK_NOTHROW(fv.ValidateAuxValues(Bind(fv, state), output));
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
    large.PrepareAuxValues(Bind(large, state), reference);
    large.PrepareAuxValues(Bind(large, state), output);
    CHECK_NOTHROW(large.ValidateAuxValues(Bind(large, state), output));
    Compare(Save(reference), output);
    auto logical = C::EquationDofs::Coupled(equation.Schema(), state);
    large.PrepareAuxValues(logical, output);
    large.ValidateAuxValues(logical, output);
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
    auto logicalState = Bind(fv, state);
    fv.Initialize(logicalState, {}, primitive);
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
        fv.PrepareAuxValues(Bind(fv, state), output);
        CHECK_NOTHROW(fv.ValidateAuxValues(Bind(fv, state), output));
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
        fv.PrepareAuxValues(Bind(fv, state), output);
        expected = Save(output);
        return fv.BeginAuxValues(Bind(fv, state), output);
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
    fv.PrepareAuxValues(Bind(fv, state), complete);
    const auto correct = *complete.cellSpectralSum;
    complete.cellSpectralSum.emplace(other, correct.Owned(), correct.Ghost());
    CHECK_THROWS(fv.ValidateAuxValues(Bind(fv, state), complete));
    complete.cellSpectralSum = correct;
    CHECK_NOTHROW(fv.ValidateAuxValues(Bind(fv, state), complete));
    auto pending = fv.BuildAuxiliaryValues({false, false, false, false, true, false, false});
    auto preparation = fv.BeginAuxValues(Bind(fv, state), pending);
    preparation.Apply();
    const auto original = *pending.cellSpectralSum;
    pending.cellSpectralSum.emplace(other, original.Owned(), original.Ghost());
    CHECK_THROWS(preparation.Finish(pending));
}
TEST_CASE("Synchronous residual completes its queued core before dependent rejection unwinds leases")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 2);
    C::DomainSettings settings;
    settings.numerics.reconstruction = "constant";
    settings.numerics.cfv.maxOrder = 1;
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::FiniteVolume fv(mesh, settings, equation);
    auto state = fv.Metrics().BuildField(5), reference = fv.Metrics().BuildField(5), result = fv.Metrics().BuildField(5);
    Initialize(fv, state, equation);
    auto logical = C::EquationDofs::Coupled(equation.Schema(), state);
    auto referenceLogical = C::EquationDofs::Coupled(equation.Schema(), reference);
    auto resultLogical = C::EquationDofs::Coupled(equation.Schema(), result);
    auto reconstruction = fv.BuildReconstructionState(logical);
    auto referenceReconstruction = fv.BuildReconstructionState(logical);
    auto work = fv.BuildWorkspace(logical), referenceWork = fv.BuildWorkspace(logical);
    R::EagerEngine baseline;
    fv.Evaluate(logical, referenceLogical, *referenceReconstruction, *referenceWork, baseline, baseline.CreateCPUStream());
    auto launcher = std::make_shared<RejectAuxiliaryDependent>();
    R::EagerEngine engine(0, std::make_shared<R::AccessTrackingContext>());
    const auto stream = engine.CreateStream(launcher);
    CHECK_THROWS(fv.Evaluate(logical, resultLogical, *reconstruction, *work, engine, stream));
    CHECK(launcher->rejected);
    CHECK(launcher->ObservedJoin());
    CHECK(launcher->coreCompleted);
    {
        auto released = state.Owned().Write();
        released.CancelUnmodified();
    }
    // The failed operation may poison whichever numerical outputs it had
    // already reserved. Durable reconstruction publications are solver-owned
    // outputs now, while the workspace is scratch; discard both before this
    // UT-only recovery. The accepted input and engine remain reusable once the
    // queued command is joined. Production collective paths abort instead.
    reconstruction = fv.BuildReconstructionState(logical);
    work = fv.BuildWorkspace(logical);
    REQUIRE_NOTHROW(fv.Evaluate(logical, resultLogical, *reconstruction, *work, engine, stream));
    const auto expected = reference.Owned().Read(), actual = result.Owned().Read();
    for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
        for (int j = 0; j < 5; ++j)
            CHECK(actual.View()(iCell, j) == expected.View()(iCell, j));
}
TEST_CASE("FV uses the same queued halo dependency sequence with inline and worker launchers")
{
    auto mesh = CauchyTest::PeriodicQuads(mpi, 8);
    C::DomainSettings settings;
    settings.numerics.reconstruction = "variational";
    settings.numerics.cfv.maxOrder = 1;
    settings.numerics.reconstructionIterations = 2;
    settings.numerics.reconstructionTolerance = 0;
    settings.numerics.positivity = false;
    settings.execution.overlap = true;
    int expectedDrains = 1;
    SUBCASE("fixed-count VR without a limiter") {}
    SUBCASE("fixed-count P3 VR with characteristic WBAP")
    {
        expectedDrains = 1; // queued characteristic producer -> WBAP -> FV return
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.cfv.intOrder = 5;
        settings.numerics.limiter = "WBAP";
        settings.numerics.characteristicLimiter = true;
    }
    SUBCASE("fixed-count P3 LVR with repeated neighborhood expansion")
    {
        expectedDrains = 2; // compound LVR return, FV return; gradients stay queued
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.cfv.intOrder = 5;
        settings.numerics.limiter = "LVR";
        settings.numerics.limitedVR.contactLayers = 2;
        settings.numerics.limitedVR.neighborLayers = 2;
    }
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::FiniteVolume fv(mesh, settings, equation);
    std::vector<DNDS::real> expected;
    std::vector<std::string> expectedSequence;
    for (const int workers : {0, 2})
    {
        auto mean = fv.Metrics().BuildField(5), residual = fv.Metrics().BuildField(5);
        Initialize(fv, mean, equation);
        auto state = Bind(fv, mean), output = Bind(fv, residual);
        auto reconstruction = fv.BuildReconstructionState(state);
        auto workspace = fv.BuildWorkspace(state);
        auto launcher = std::make_shared<RecordingAuxiliaryLauncher>(workers);
        R::EagerEngine engine(0, std::make_shared<R::AccessTrackingContext>());
        const auto stream = engine.CreateStream(launcher);
        fv.Evaluate(state, output, *reconstruction, *workspace, engine, stream, .25);
        // Fixed-count VR never drains between sweeps; P3 WBAP never drains
        // between orders. LVR does not drain between neighborhood layers.
        // Only the declared public/CPU preparation boundaries count here.
        // Portable MPI producer-prefix observations are a separate backend
        // contract and are deliberately not disguised as full-engine drains.
        CHECK(std::count(launcher->trace.begin(), launcher->trace.end(), "drain") == expectedDrains);
        std::vector<DNDS::real> values;
        {
            const auto read = residual.Owned().Read();
            const auto view = read.View<R::UniformTableRows>();
            for (DNDS::index iCell = 0; iCell < residual.Owned().RowCount(); ++iCell)
                for (DNDS::index j = 0; j < 5; ++j)
                    values.push_back(view(iCell, j));
        }
        // Prefix observations can vary with progress timing; submitted
        // numerical commands and stream dependencies cannot.
        std::vector<std::string> sequence;
        for (const auto &entry : launcher->trace)
            if (entry == "submit" || entry == "wait")
                sequence.push_back(entry);
        // A leading dependency imports a possibly queued stage producer into
        // the transfer lane. It is distinct from the later halo dependency:
        // require submitted work on both sides of that later boundary.
        const auto firstSubmit = std::find(sequence.begin(), sequence.end(), "submit");
        REQUIRE(firstSubmit != sequence.end());
        const auto wait = std::find(firstSubmit + 1, sequence.end(), "wait");
        REQUIRE(wait != sequence.end());
        CHECK(std::find(wait + 1, sequence.end(), "submit") != sequence.end());
        if (workers == 0)
        {
            expected = values;
            expectedSequence = sequence;
        }
        else
        {
            CHECK(values == expected);
            CHECK(sequence == expectedSequence);
        }
        auto released = mean.Ghost().Write();
        released.CancelUnmodified();
    }
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
