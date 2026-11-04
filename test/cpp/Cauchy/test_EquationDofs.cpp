#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/State/EquationDofs.hpp"
#include "Cauchy/Physics/EquationSystem.hpp"
#include "Cauchy/Implicit/LinearSolver.hpp"
#include "Cauchy/Config/SimulationConfig.hpp"
#include "DNDS/Runtime/Communication/RowTransfer.hpp"
#include "Solver/FieldTimeStepper.hpp"
#include "Cauchy/State/EquationTimeStepper.hpp"
#include "Cauchy/State/EquationGMRES.hpp"
#include <numeric>
#include <optional>

namespace C = DNDS::Cauchy;
namespace A = DNDS::Algebra;
namespace R = DNDS::Runtime;
using I = DNDS::index;
static DNDS::MPIInfo mpi;
namespace
{
    C::EquationSchema Schema(int flow)
    {
        C::EquationSchema schema{"mock-coupled", "mock-energy", "identity", {}};
        C::EquationBlockSchema block;
        block.name = "flow";
        for (const auto *name : flow == 4 ? std::vector<const char *>{"u", "v", "w", "p"}
                                          : std::vector<const char *>{"rho", "rhou", "rhov", "rhow", "rhop"})
            block.components.push_back({name, "mock-units", "flow"});
        block.vectorStarts = {flow == 4 ? 0 : 1};
        schema.blocks.push_back(std::move(block));
        for (const auto &[name, count] : std::vector<std::pair<std::string, int>>{{"rans", flow == 4 ? 1 : 2}, {"species", flow == 4 ? 12 : 14}})
        {
            C::EquationBlockSchema extra;
            extra.name = name;
            for (int j = 0; j < count; ++j)
                extra.components.push_back({name + std::to_string(j), "mock-units", name});
            schema.blocks.push_back(std::move(extra));
        }
        return schema;
    }
    I Width(const C::EquationSchema &schema)
    {
        I count = 0;
        for (const auto &block : schema.blocks)
            count += I(block.components.size());
        return count;
    }
    R::RowLayout Layout(I width, int layout)
    {
        if (layout == 1)
            return R::RowLayout::ColMajor(0, width, {8, 1});
        if (layout == 2)
            return R::RowLayout::TiledColMajor(0, width, 32);
        return R::RowLayout::Dense(0, width, width + 3);
    }
    struct Fixture
    {
        std::shared_ptr<R::CommunicationContext> communication;
        int active;
        I rows, ghosts;
        A::DofSpace space;
        R::RowTransferPlan route;
        static R::RowTransferPlan Route(const std::shared_ptr<R::CommunicationContext> &context, int active, I rows, I ghosts)
        {
            std::vector<std::vector<I>> send(mpi.size), receive(mpi.size);
            if (mpi.rank < active)
            {
                send[(mpi.rank + 1) % active] = {rows - 2, rows - 1};
                receive[(mpi.rank + active - 1) % active] = {0, 1};
            }
            return R::RowTransferPlan::Known(context, rows, ghosts, send, receive);
        }
        Fixture()
            : communication(std::make_shared<R::CommunicationContext>(mpi.comm)), active(mpi.size > 2 ? mpi.size - 1 : mpi.size),
              rows(mpi.rank < active ? 35 : 0), ghosts(mpi.rank < active ? 2 : 0),
              space(communication, size_t(rows), size_t(ghosts), "semantic DOF ring"), route(Route(communication, active, rows, ghosts)) {}
        C::EquationDofs Build(const C::EquationSchema &schema, int layout, bool split) const
        {
            if (!split)
                return C::EquationDofs::Coupled(schema, A::DofField<double>(space, Layout(Width(schema), layout)), "packed");
            A::DofBundle fields(space);
            for (const auto &block : schema.blocks)
                fields.Add(block.name, A::DofField<double>(space, Layout(I(block.components.size()), layout)));
            return C::EquationDofs::Separate(schema, std::move(fields));
        }
        R::TransferBinding Bind(A::DofField<double> &field) const
        {
            return R::PayloadPlan(route, field.Owned().Layout(), field.Ghost().Layout(), R::ElementSchema::For<double>())
                .Bind(field.Owned(), field.Ghost());
        }
    };
    double Value(int owner, I row, I component) { return 1 + 100 * owner + 2 * row + .125 * component; }
    void Fill(C::EquationDofs &state)
    {
        auto write = state.WriteOwned();
        I first = 0;
        for (I iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
        {
            const auto out = write.Block(iBlock);
            for (I iRow = 0; iRow < out.RowCount(); ++iRow)
                for (I j = 0; j < out.ComponentNumber(); ++j)
                    out(iRow, j) = Value(mpi.rank, iRow, first + j);
            first += out.ComponentNumber();
        }
    }
    void CheckValues(const C::EquationDofs &state, double scale = 1, double shift = 0)
    {
        const auto read = state.ReadOwned();
        I first = 0;
        for (I iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
        {
            const auto in = read.Block(state.Schema().blocks[iBlock].name);
            for (I iRow = 0; iRow < in.owned.RowCount(); ++iRow)
                for (I j = 0; j < in.ComponentNumber(); ++j)
                    CHECK(in(iRow, j) == scale * Value(mpi.rank, iRow, first + j) + shift);
            first += in.ComponentNumber();
        }
    }
    void Pointwise(const C::EquationDofs &input, C::EquationDofs &output)
    {
        input.RequireCompatible(output);
        const auto read = input.ReadOwned();
        auto write = output.WriteOwned();
        for (I iBlock = 0; iBlock < input.BlockNumber(); ++iBlock)
        {
            const auto in = read.Block(iBlock).owned;
            const auto out = write.Block(iBlock);
            for (I iRow = 0; iRow < out.RowCount(); ++iRow)
                for (I j = 0; j < out.ComponentNumber(); ++j)
                    out[iRow][j] = 3 * in[iRow][j] - 2;
        }
    }
}
static_assert(std::is_trivially_copyable_v<C::EquationBlockArrayView<const double>>);
static_assert(std::is_trivially_copyable_v<C::EquationBlockReadView>);
TEST_CASE("Logical flow4/5 with RANS and many species maps exactly across physical layouts and storage groups")
{
    Fixture fixture;
    for (int flow : {4, 5})
        for (int layout = 0; layout < 3; ++layout)
            for (bool split : {false, true})
            {
                const auto schema = Schema(flow);
                auto state = fixture.Build(schema, layout, split);
                Fill(state);
                CheckValues(state);
                CHECK(state.ComponentNumber() == (flow == 4 ? 17 : 21));
                CHECK(state.BlockNumber() == 3);
                CHECK(state.PhysicalFieldNumber() == (split ? 3 : 1));
                I componentOffset = 0;
                for (I iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                {
                    const auto location = state.LocateBlock(schema.blocks[iBlock].name);
                    CHECK(location.physicalField == (split ? iBlock : 0));
                    CHECK(location.firstComponent == (split ? 0 : componentOffset));
                    CHECK(location.componentNumber == I(schema.blocks[iBlock].components.size()));
                    componentOffset += location.componentNumber;
                }
                // Independent native full-field oracle catches a segment view
                // that rewrites a tiled parent's physical component count.
                I logicalFirst = 0;
                for (I iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                {
                    const auto &field = state.PhysicalField(iField);
                    const auto read = field.Owned().Read();
                    const auto view = read.View<R::UniformTableRows>();
                    for (I iRow = 0; iRow < fixture.rows; ++iRow)
                        for (I j = 0; j < view.Width(); ++j)
                            CHECK(view(iRow, j) == Value(mpi.rank, iRow, logicalFirst + j));
                    logicalFirst += view.Width();
                }
                auto output = state.RebindFields(state.Fields().Clone());
                Pointwise(state, output);
                CheckValues(output, 3, -2);
                CheckValues(state);
                if (split)
                    CHECK_THROWS(state.CoupledField());
                else
                    CHECK(state.CoupledField().Width() == size_t(state.ComponentNumber()));
                {
                    auto writer = output.WriteOwned();
                    const auto selected = writer.Block("rans");
                    for (I iRow = 0; iRow < selected.RowCount(); ++iRow)
                        for (I j = 0; j < selected.ComponentNumber(); ++j)
                            selected(iRow, j) = -7;
                    // Each physical writer is acquired once even when all
                    // three logical ranges share one allocation.
                    const auto flowView = writer.Block("flow");
                    if (flow == 5)
                    {
                        const auto fixed = flowView.Fixed<5>();
                        static_assert(decltype(fixed[0])::row_width == 5);
                    }
                    else
                        CHECK_THROWS(flowView.Fixed<5>());
                }
                const auto read = output.ReadOwned();
                I first = 0;
                for (I iBlock = 0; iBlock < output.BlockNumber(); ++iBlock)
                {
                    const auto block = read.Block(iBlock).owned;
                    for (I iRow = 0; iRow < fixture.rows; ++iRow)
                        for (I j = 0; j < block.ComponentNumber(); ++j)
                            CHECK(block(iRow, j) == (iBlock == 1 ? -7 : 3 * Value(mpi.rank, iRow, first + j) - 2));
                    first += block.ComponentNumber();
                }
            }
}
TEST_CASE("Hybrid equation blocks expose physical grouping and rebind across parent layouts")
{
    Fixture fixture;
    A::DofBundle fields(fixture.space);
    fields.Add("chemistry", A::DofField<double>(fixture.space, Layout(14, 1)));
    fields.Add("fluid", A::DofField<double>(fixture.space, Layout(7, 2)));
    C::EquationDofs state(Schema(5), std::move(fields), {{"flow", "fluid", 0}, {"rans", "fluid", 5}, {"species", "chemistry", 0}});
    Fill(state);
    const auto flow = state.LocateBlock("flow"), rans = state.LocateBlock("rans"), species = state.LocateBlock("species");
    CHECK(flow.physicalField == 1);
    CHECK(rans.physicalField == flow.physicalField);
    CHECK(rans.firstComponent == 5);
    CHECK(species.physicalField == 0);
    CHECK(state.PhysicalField(flow.physicalField).Width() == 7);
    CHECK(state.PhysicalField(species.physicalField).Width() == 14);
    A::DofBundle stageFields(fixture.space);
    stageFields.Add("chemistry", A::DofField<double>(fixture.space, Layout(14, 2)));
    stageFields.Add("fluid", A::DofField<double>(fixture.space, Layout(7, 0)));
    auto stage = state.RebindFields(std::move(stageFields));
    Pointwise(state, stage);
    CheckValues(stage, 3, -2);
    CheckValues(state);
    CHECK(stage.LocateBlock("rans").firstComponent == 5);
}
TEST_CASE("Owned-only pointwise algebra and reductions overlap direct halo writes without regional splitting")
{
    Fixture fixture;
    for (int layout = 0; layout < 3; ++layout)
        for (bool split : {false, true})
        {
            auto state = fixture.Build(Schema(5), layout, split);
            Fill(state);
            auto output = state.Clone();
            std::vector<R::TransferBinding> bindings;
            bindings.reserve(size_t(state.PhysicalFieldNumber()));
            for (I iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                bindings.push_back(fixture.Bind(state.PhysicalField(iField)));
            std::vector<R::TransferOperation> transfers;
            transfers.reserve(bindings.size());
            for (auto &binding : bindings)
                transfers.push_back(binding.Begin());
            Pointwise(state, output);
            CheckValues(output, 3, -2);
            if (fixture.ghosts)
            {
                CHECK_THROWS(state.ReadWithGhost());
                CHECK_THROWS(state.WriteOwned()); // Direct sends still read owned storage.
            }
            double sum = 0;
            {
                const auto read = state.ReadOwned();
                for (I iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                {
                    const auto v = read.Block(iBlock).owned;
                    for (I iRow = 0; iRow < v.RowCount(); ++iRow)
                        for (I j = 0; j < v.ComponentNumber(); ++j)
                            sum += v(iRow, j) * v(iRow, j);
                }
            }
            MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, fixture.communication->Communicator());
            double expected = 0;
            for (int owner = 0; owner < fixture.active; ++owner)
                for (I iRow = 0; iRow < 35; ++iRow)
                    for (I j = 0; j < state.ComponentNumber(); ++j)
                        expected += Value(owner, iRow, j) * Value(owner, iRow, j);
            CHECK(sum == doctest::Approx(expected).epsilon(1e-13));
            for (auto &transfer : transfers)
                transfer.Wait();
            const auto read = state.ReadWithGhost();
            I first = 0;
            for (I iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
            {
                const auto in = read.Block(iBlock);
                for (I iGhost = 0; iGhost < fixture.ghosts; ++iGhost)
                    for (I j = 0; j < in.ComponentNumber(); ++j)
                        CHECK(in(fixture.rows + iGhost, j) == Value((mpi.rank + fixture.active - 1) % fixture.active, 33 + iGhost, first + j));
                first += in.ComponentNumber();
            }
        }
}
TEST_CASE("Semantic validation and writer reservation reject bad bindings without partial publication")
{
    Fixture fixture;
    const auto schema = Schema(5);
    const auto build = [&](C::EquationSchema requested, std::vector<C::EquationDofBinding> bindings, I width = 21)
    {
        A::DofBundle fields(fixture.space);
        fields.Add("packed", A::DofField<double>(fixture.space, Layout(width, 2)));
        return C::EquationDofs(std::move(requested), std::move(fields), std::move(bindings));
    };
    const std::vector<C::EquationDofBinding> ordinary{{"flow", "packed", 0}, {"rans", "packed", 5}, {"species", "packed", 7}};
    auto malformed = ordinary;
    malformed[1].first = 4;
    CHECK_THROWS(build(schema, malformed));
    malformed = ordinary;
    malformed[0].first = -1;
    CHECK_THROWS(build(schema, malformed));
    malformed = ordinary;
    malformed[0].first = 20;
    CHECK_THROWS(build(schema, malformed));
    malformed = ordinary;
    malformed[2].field = "missing";
    CHECK_THROWS(build(schema, malformed));
    malformed = ordinary;
    malformed[2].block = "flow";
    CHECK_THROWS(build(schema, malformed));
    CHECK_THROWS(build(schema, ordinary, 22));
    auto badSchema = schema;
    badSchema.blocks[0].vectorStarts = {4};
    CHECK_THROWS(build(badSchema, ordinary));
    badSchema = schema;
    badSchema.blocks[1].name = "flow";
    CHECK_THROWS(build(badSchema, ordinary));
    auto permuted = build(schema, {{"species", "packed", 0}, {"flow", "packed", 14}, {"rans", "packed", 19}});
    Fill(permuted);
    CheckValues(permuted);
    CHECK_THROWS(permuted.CoupledField());
    {
        const auto read = permuted.PhysicalField(0).Owned().Read();
        const auto values = read.View<R::UniformTableRows>();
        for (I iRow = 0; iRow < fixture.rows; ++iRow)
        {
            for (I j = 0; j < 14; ++j)
                CHECK(values(iRow, j) == Value(mpi.rank, iRow, 7 + j));
            for (I j = 0; j < 5; ++j)
                CHECK(values(iRow, 14 + j) == Value(mpi.rank, iRow, j));
            for (I j = 0; j < 2; ++j)
                CHECK(values(iRow, 19 + j) == Value(mpi.rank, iRow, 5 + j));
        }
    }
    auto state = fixture.Build(schema, 1, true);
    Fill(state);
    std::vector<uint64_t> versions;
    for (I iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
        versions.push_back(state.PhysicalField(iField).Owned().Version());
    {
        const auto held = state.PhysicalField(2).Owned().Read();
        CHECK_THROWS(state.WriteOwned()); // Earlier field reservations must cancel unchanged.
    }
    for (I iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
        CHECK(state.PhysicalField(iField).Owned().Version() == versions[iField]);
    CheckValues(state);
    CHECK_THROWS(state.ReadOwned().Block("missing"));
    {
        auto write = state.WriteOwned();
        CHECK_THROWS(write.Block(3));
        write.CancelUnmodified();
    }
    const auto model = C::EquationSystem::Resolve(C::EquationSettings{});
    auto pressure = model.Schema();
    pressure.blocks[0].components.back().name = "rhop";
    auto pressureState = C::EquationDofs::Coupled(pressure, A::DofField<double>(fixture.space, 5));
    CHECK(pressureState.ComponentNumber() == 5);
    CHECK_THROWS(pressureState.RequireSchema(model.Schema())); // Same width, different equation meaning.
    CHECK_THROWS(model.ValidateStateSchema(pressureState));
    auto ordinaryState = C::EquationDofs::Coupled(model.Schema(), A::DofField<double>(fixture.space, 5));
    CHECK_NOTHROW(model.ValidateStateSchema(ordinaryState));
    CHECK(model.ResolveKernelProfile(R::Device::Host()) == C::EquationKernelProfile::CompressiblePerfectGasCPU);
    CHECK_THROWS(model.ResolveKernelProfile(R::Device::CUDA(0))); // No CPU fallback for a missing device kernel.
}
TEST_CASE("Retained scopes preserve allocation lifetime and storage replacement requires explicit rebinding")
{
    Fixture fixture;
    auto retained = [&]
    {
        auto temporary = fixture.Build(Schema(5), 2, false);
        Fill(temporary);
        return temporary.ReadOwned();
    }();
    // Churn equally sized allocations after the state owner has disappeared.
    std::vector<A::DofField<double>> churn;
    for (int i = 0; i < 8; ++i)
    {
        churn.emplace_back(fixture.space, Layout(21, 2));
        A::Fill(churn.back(), -999.);
    }
    const auto flow = retained.Block("flow").owned.Fixed<5>();
    for (I iRow = 0; iRow < fixture.rows; ++iRow)
        for (I j = 0; j < 5; ++j)
            CHECK(flow[iRow][j] == Value(mpi.rank, iRow, j));
    auto state = fixture.Build(Schema(5), 0, false);
    Fill(state);
    auto &physical = state.CoupledField();
    const auto before = state.ReadOwned();
    physical.Owned() = R::Array<double>(physical.Owned().Layout());
    CHECK_THROWS(state.Validate());
    const auto old = before.Block("species").owned;
    for (I iRow = 0; iRow < fixture.rows; ++iRow)
        for (I j = 0; j < 14; ++j)
            CHECK(old(iRow, j) == Value(mpi.rank, iRow, 7 + j));
}
TEST_CASE("Coupled matrix action ILU and left GMRES preserve separate and permuted logical equation fields")
{
    Fixture fixture;
    const auto schema = Schema(5);
    constexpr I nv = 21;
    const auto diagonal = [nv](I row)
    {
        Eigen::MatrixXd value(nv, nv);
        for (I j = 0; j < nv; ++j)
            for (I i = 0; i < nv; ++i)
                value(i, j) = (i == j ? 4 + .001 * row + .002 * i : 0) + .001 * (i + 1) * (j + 1) / double(nv * nv);
        return value;
    };
    const auto off = [nv](I row, I column)
    {
        Eigen::MatrixXd value(nv, nv);
        for (I j = 0; j < nv; ++j)
            for (I i = 0; i < nv; ++i)
                value(i, j) = (i == j ? -.03 - .00001 * row : 0) + .0001 * (i + 1) * (column + j + 1) / double(nv * nv);
        return value;
    };
    const auto vector = [nv](int owner, I row)
    {
        Eigen::VectorXd value(nv);
        for (I j = 0; j < nv; ++j)
            value[j] = Value(owner, row, j);
        return value;
    };
    std::vector<std::vector<I>> columns(size_t(fixture.rows));
    std::vector<I> core;
    for (I iRow = 0; iRow < fixture.rows; ++iRow)
    {
        if (iRow > 0)
            columns[iRow].push_back(iRow - 1);
        if (iRow + 1 < fixture.rows)
            columns[iRow].push_back(iRow + 1);
        if (iRow + 2 >= fixture.rows)
            columns[iRow].push_back(fixture.rows + iRow - (fixture.rows - 2));
        else
            core.push_back(iRow);
    }
    C::LinearProblemContext problem(fixture.space, nv, columns, core, C::DiagonalStorage::Block);
    {
        auto d = problem.Diagonal().Write();
        auto b = problem.OffDiagonal().Write();
        const auto matrices = A::MakeEigenRowMatrices(d.View<R::TableRows>(), nv, nv);
        for (I iRow = 0; iRow < fixture.rows; ++iRow)
        {
            matrices[iRow] = diagonal(iRow);
            for (I j = 0; j < I(columns[iRow].size()); ++j)
                b.View()(iRow, j) = off(iRow, columns[iRow][j]);
        }
    }
    problem.PrepareILUStructure(-1, false);
    problem.FactorizeILU();
    for (int layout = 0; layout < 3; ++layout)
    {
        auto state = fixture.Build(schema, layout, layout != 0);
        if (layout == 2)
        {
            A::DofBundle fields(fixture.space);
            fields.Add("packed", A::DofField<double>(fixture.space, Layout(nv, 2)));
            state = C::EquationDofs(schema, std::move(fields), {{"flow", "packed", 14}, {"rans", "packed", 19}, {"species", "packed", 0}});
        }
        Fill(state);
        auto result = fixture.Build(schema, (layout + 1) % 3, true);
        auto rhs = C::EquationDofOperations::NewVector(result), localRhs = C::EquationDofOperations::NewVector(result);
        auto forward = C::EquationDofOperations::NewVector(result), solved = C::EquationDofOperations::NewVector(result);
        std::vector<R::TransferBinding> halo;
        for (I iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            halo.push_back(fixture.Bind(state.PhysicalField(iField)));
        std::vector<R::TransferOperation> operations;
        for (auto &binding : halo)
            operations.push_back(binding.Begin());
        problem.Apply(state, result, C::LinearRegion::Core);
        for (auto &operation : operations)
            operation.Wait();
        problem.Apply(state, result, C::LinearRegion::HaloDependent);
        {
            const auto read = result.ReadOwned();
            const auto layoutRead = read.ComponentLayout();
            const auto values = layoutRead.View();
            auto write = rhs.WriteOwned(), localWrite = localRhs.WriteOwned();
            const auto rhsLayout = write.ComponentLayout(), localLayout = localWrite.ComponentLayout();
            const auto b = rhsLayout.View(), local = localLayout.View();
            for (I iRow = 0; iRow < fixture.rows; ++iRow)
            {
                Eigen::VectorXd expected = diagonal(iRow) * vector(mpi.rank, iRow);
                Eigen::VectorXd localExpected = expected;
                for (const auto column : columns[iRow])
                {
                    if (column < fixture.rows)
                    {
                        const Eigen::VectorXd addition = off(iRow, column) * vector(mpi.rank, column);
                        expected += addition;
                        localExpected += addition;
                    }
                    else
                    {
                        const int previous = (mpi.rank + fixture.active - 1) % fixture.active;
                        expected.noalias() += off(iRow, column) * vector(previous, fixture.rows - 2 + column - fixture.rows);
                    }
                }
                for (I j = 0; j < nv; ++j)
                {
                    CHECK(values(iRow, j) == doctest::Approx(expected[j]).epsilon(2e-13));
                    b(iRow, j) = expected[j];
                    local(iRow, j) = localExpected[j];
                }
            }
        }
        problem.ApplyILU(localRhs, forward, solved);
        {
            const auto read = solved.ReadOwned();
            const auto view = read.ComponentLayout();
            for (I iRow = 0; iRow < fixture.rows; ++iRow)
                for (I j = 0; j < nv; ++j)
                    CHECK(view.View()(iRow, j) == doctest::Approx(Value(mpi.rank, iRow, j)).epsilon(2e-12));
        }
        C::EquationGMRES gmres(state, 8);
        struct Input
        {
            C::EquationDofs *state;
            std::vector<R::TransferBinding> halo;
        };
        std::vector<Input> inputs;
        gmres.VisitOperatorInputs([&](auto &input)
                                  {
            Input value{&input,{}};
            for (I iField = 0; iField < input.PhysicalFieldNumber(); ++iField) value.halo.push_back(fixture.Bind(input.PhysicalField(iField)));
            inputs.push_back(std::move(value)); });
        const auto apply = [&](auto &input, auto &output)
        {
            const auto found = std::find_if(inputs.begin(), inputs.end(), [&](const auto &v)
                                            { return v.state == &input; });
            REQUIRE(found != inputs.end());
            std::vector<R::TransferOperation> active;
            for (auto &binding : found->halo)
                active.push_back(binding.Begin());
            problem.Apply(input, output, C::LinearRegion::Core);
            for (auto &operation : active)
                operation.Wait();
            problem.Apply(input, output, C::LinearRegion::HaloDependent);
        };
        const auto products = [&](const auto &input, const auto &probes)
        {
            auto result = C::EquationDofOperations::LocalProducts(input, probes);
            MPI_Allreduce(MPI_IN_PLACE, result.data(), int(result.size()), MPI_DOUBLE, MPI_SUM, mpi.comm);
            return result;
        };
        const auto report = gmres.Solve(rhs, apply, [&](const auto &input, auto &output)
                                        { problem.ApplyILU(input, forward, output); }, products, {40, 1e-12, 1e-10});
        CHECK(report.converged);
        CHECK(report.usable);
        {
            const auto read = gmres.Solution().ReadOwned();
            const auto layoutRead = read.ComponentLayout();
            const auto values = layoutRead.View();
            for (I iRow = 0; iRow < fixture.rows; ++iRow)
                for (I j = 0; j < nv; ++j)
                    CHECK(values(iRow, j) == doctest::Approx(Value(mpi.rank, iRow, j)).epsilon(2e-10));
        }
        if (mpi.rank == 0)
        {
            const auto generation = forward.PhysicalField(0).Owned().Version();
            const auto outputGeneration = solved.PhysicalField(0).Owned().Version();
            auto pin = solved.PhysicalField(solved.PhysicalFieldNumber() - 1).Owned().Read();
            CHECK_THROWS(problem.ApplyILU(localRhs, forward, solved));
            CHECK(forward.PhysicalField(0).Owned().Version() == generation);
            CHECK(solved.PhysicalField(0).Owned().Version() == outputGeneration);
        }
    }
}
TEST_CASE("The same BDF and coupled block solve interface advances all flow RANS and species components")
{
    Fixture fixture;
    for (int flow : {4, 5})
        for (int layout = 0; layout < 3; ++layout)
        {
            auto state = fixture.Build(Schema(flow), layout, false);
            Fill(state);
            auto trial = state.Clone();
            const I nv = state.ComponentNumber();
            std::vector<double> a(size_t(nv), 0.);
            double square = 0;
            for (I j = 0; j < nv; ++j)
            {
                a[j] = double(j + 1) / double(nv);
                square += a[j] * a[j];
            }
            DNDS::ODE::FieldTimeStepper stepper(state.CoupledField(), "bdf1");
            I solves = 0;
            const auto solve = [&](auto &candidate, const auto &source, double mass, double scale, double, int)
            {
                ++solves;
                auto problem = std::make_shared<C::LinearProblemContext>(fixture.space, nv, std::vector<std::vector<I>>(size_t(fixture.rows)),
                                                                         std::vector<I>{}, C::DiagonalStorage::Block);
                {
                    auto diagonal = problem->Diagonal().Write();
                    const auto values = diagonal.View<R::UniformTableRows>();
                    for (I iRow = 0; iRow < fixture.rows; ++iRow)
                        for (I j = 0; j < nv; ++j)
                            for (I i = 0; i < nv; ++i)
                                values(iRow, i + nv * j) = (i == j ? mass + scale : 0) + scale * a[i] * a[j];
                }
                problem->InvertDiagonal();
                C::LinearSolveSettings control;
                control.maxIterations = 1;
                control.relativeTolerance = 1e-12;
                control.absoluteTolerance = 1e-10;
                C::LinearSolver solver(problem, source, [&](auto &field)
                                       { return fixture.Bind(field); }, control);
                const auto report = solver.Solve(source);
                CHECK(report.converged);
                CHECK(report.iterations == 1);
                A::Copy(candidate, solver.Solution());
                return report.converged;
            };
            // F(U)=-U-a(a.U), coupling every logical segment. BDF1 does not
            // need a separate explicit rate call; its stage solver sees all nv.
            const auto evaluate = [](auto &, auto &, double, int)
            { FAIL("BDF1 must not request an explicit rate"); };
            constexpr double dt = .025;
            for (int step = 0; step < 2; ++step)
            {
                REQUIRE(stepper.Prepare(state.CoupledField(), trial.CoupledField(), step * dt, dt, evaluate, solve));
                {
                    const auto initial = state.ReadOwned(), result = trial.ReadOwned();
                    for (I iRow = 0; iRow < fixture.rows; ++iRow)
                    {
                        double dot = 0;
                        I first = 0;
                        for (I iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                        {
                            const auto values = initial.Block(iBlock).owned;
                            for (I j = 0; j < values.ComponentNumber(); ++j)
                                dot += a[first + j] * values(iRow, j);
                            first += values.ComponentNumber();
                        }
                        first = 0;
                        for (I iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                        {
                            const auto x = initial.Block(iBlock).owned, y = result.Block(iBlock).owned;
                            for (I j = 0; j < x.ComponentNumber(); ++j)
                            {
                                // Independent Sherman-Morrison oracle, not the
                                // native block inverse or the stage-solver result.
                                const double expected = x(iRow, j) / (1 + dt) - dt * a[first + j] * dot / ((1 + dt) * (1 + dt + dt * square));
                                CHECK(y(iRow, j) == doctest::Approx(expected).epsilon(2e-12));
                            }
                            first += x.ComponentNumber();
                        }
                    }
                }
                // Immediate algebra requires idle inputs; the engine-aware
                // variants support compatible active readers.
                A::Copy(state.CoupledField(), trial.CoupledField());
                stepper.Commit();
            }
            CHECK(solves == 2);
        }
}
TEST_CASE("One time driver advances coupled and separate equation storage with identical coupled stage mathematics")
{
    Fixture fixture;
    constexpr I nv = 21;
    std::array<double, nv> a{};
    double square = 0;
    for (I j = 0; j < nv; ++j)
    {
        a[j] = double(j + 1) / nv;
        square += a[j] * a[j];
    }
    const auto evaluate = [&](C::EquationDofs &state, C::EquationDofs &rate, double, int)
    {
        const auto input = state.ReadOwned();
        auto output = rate.WriteOwned();
        std::array<C::EquationBlockArrayView<const double>, 3> u;
        std::array<C::EquationBlockArrayView<double>, 3> f;
        const I offsets[]{0, 5, 7};
        for (I iBlock = 0; iBlock < 3; ++iBlock)
        {
            u[iBlock] = input.Block(iBlock).owned;
            f[iBlock] = output.Block(iBlock);
        }
        for (I iRow = 0; iRow < fixture.rows; ++iRow)
        {
            double projection = 0;
            for (I iBlock = 0; iBlock < 3; ++iBlock)
                for (I j = 0; j < u[iBlock].ComponentNumber(); ++j)
                    projection += a[offsets[iBlock] + j] * u[iBlock](iRow, j);
            for (I iBlock = 0; iBlock < 3; ++iBlock)
                for (I j = 0; j < u[iBlock].ComponentNumber(); ++j)
                    f[iBlock](iRow, j) = -u[iBlock](iRow, j) - a[offsets[iBlock] + j] * projection;
        }
    };
    const auto solve = [&](C::EquationDofs &state, const C::EquationDofs &source, double mass, double scale, double, int)
    {
        const auto input = source.ReadOwned();
        auto output = state.WriteOwned();
        std::array<C::EquationBlockArrayView<const double>, 3> b;
        std::array<C::EquationBlockArrayView<double>, 3> u;
        const I offsets[]{0, 5, 7};
        for (I iBlock = 0; iBlock < 3; ++iBlock)
        {
            b[iBlock] = input.Block(iBlock).owned;
            u[iBlock] = output.Block(iBlock);
        }
        const double diagonal = mass + scale;
        for (I iRow = 0; iRow < fixture.rows; ++iRow)
        {
            double projection = 0;
            for (I iBlock = 0; iBlock < 3; ++iBlock)
                for (I j = 0; j < b[iBlock].ComponentNumber(); ++j)
                    projection += a[offsets[iBlock] + j] * b[iBlock](iRow, j);
            for (I iBlock = 0; iBlock < 3; ++iBlock)
                for (I j = 0; j < b[iBlock].ComponentNumber(); ++j)
                    u[iBlock](iRow, j) = b[iBlock](iRow, j) / diagonal - scale * a[offsets[iBlock] + j] * projection / (diagonal * (diagonal + scale * square));
        }
        return true;
    };
    using Field = A::DofField<double>;
    const auto nativeRate = [&](Field &state, Field &rate, double, int)
    {
        const auto read = state.Owned().Read();
        auto write = rate.Owned().Write();
        const auto u = read.View<R::UniformTableRows>();
        const auto f = write.View<R::UniformTableRows>();
        for (I iRow = 0; iRow < fixture.rows; ++iRow)
        {
            double projection = 0;
            for (I j = 0; j < nv; ++j)
                projection += a[j] * u(iRow, j);
            for (I j = 0; j < nv; ++j)
                f(iRow, j) = -u(iRow, j) - a[j] * projection;
        }
    };
    const auto nativeSolve = [&](Field &state, const Field &source, double mass, double scale, double, int)
    {
        const auto read = source.Owned().Read();
        auto write = state.Owned().Write();
        const auto b = read.View<R::UniformTableRows>();
        const auto u = write.View<R::UniformTableRows>();
        const double diagonal = mass + scale;
        for (I iRow = 0; iRow < fixture.rows; ++iRow)
        {
            double projection = 0;
            for (I j = 0; j < nv; ++j)
                projection += a[j] * b(iRow, j);
            for (I j = 0; j < nv; ++j)
                u(iRow, j) = b(iRow, j) / diagonal - scale * a[j] * projection / (diagonal * (diagonal + scale * square));
        }
        return true;
    };
    const auto converged = [&](const C::EquationDofs &h, const C::EquationDofs &i, I)
    {
        const auto rh = h.ReadOwned(), ri = i.ReadOwned();
        double error = 0;
        for (I iBlock = 0; iBlock < 3; ++iBlock)
        {
            const auto hv = rh.Block(iBlock).owned, iv = ri.Block(iBlock).owned;
            for (I iRow = 0; iRow < fixture.rows; ++iRow)
                for (I j = 0; j < hv.ComponentNumber(); ++j)
                    error = std::max({error, std::abs(hv(iRow, j)), std::abs(iv(iRow, j))});
        }
        MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
        return error < 1e-8;
    };
    const auto nativeConverged = [&](const Field &h, const Field &i, I)
    {
        const auto rh = h.Owned().Read(), ri = i.Owned().Read();
        double error = 0;
        const auto hv = rh.View<R::UniformTableRows>(), iv = ri.View<R::UniformTableRows>();
        for (I iRow = 0; iRow < fixture.rows; ++iRow)
            for (I j = 0; j < nv; ++j)
                error = std::max({error, std::abs(hv(iRow, j)), std::abs(iv(iRow, j))});
        MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
        return error < 1e-8;
    };
    for (const std::string method : {"bdf1", "bdf2", "trapz", "esdirk2", "esdirk3", "esdirk4", "ditr"})
    {
        INFO(method);
        A::DofBundle fields(fixture.space);
        fields.Add("flow", Field(fixture.space, Layout(5, 0)));
        fields.Add("rans", Field(fixture.space, Layout(2, 2)));
        fields.Add("species", Field(fixture.space, Layout(14, 1)));
        auto state = C::EquationDofs::Separate(Schema(5), std::move(fields));
        Fill(state);
        auto candidate = C::EquationDofOperations::NewVector(state);
        auto reference = fixture.Build(Schema(5), 0, false);
        Fill(reference);
        auto nativeCandidate = DNDS::ODE::FieldTimeOperations::NewVector(reference.CoupledField());
        C::EquationTimeStepper driver(state, method, {}, {}, [&](auto &)
                                      { return C::EquationDITR::StageOperator{evaluate, solve, {}}; });
        DNDS::ODE::FieldTimeStepper oracle(reference.CoupledField(), method, {}, {}, [&](auto &)
                                           { return DNDS::ODE::FieldDITR::StageOperator{nativeRate, nativeSolve, {}}; });
        const auto compare = [&]
        {
            const auto read = candidate.ReadOwned();
            const auto flat = nativeCandidate.Owned().Read();
            const auto values = flat.View<R::UniformTableRows>();
            I first = 0;
            for (I iBlock = 0; iBlock < 3; ++iBlock)
            {
                const auto block = read.Block(iBlock).owned;
                for (I iRow = 0; iRow < fixture.rows; ++iRow)
                    for (I j = 0; j < block.ComponentNumber(); ++j)
                        CHECK(block(iRow, j) == values(iRow, first + j));
                first += block.ComponentNumber();
            }
        };
        double time = 0;
        for (double dt : {.02, .03})
        {
            REQUIRE(driver.Prepare(state, candidate, time, dt, evaluate, solve, converged));
            REQUIRE(oracle.Prepare(reference.CoupledField(), nativeCandidate, time, dt, nativeRate, nativeSolve, nativeConverged));
            compare();
            C::EquationDofOperations::Copy(state, candidate);
            driver.Commit();
            A::Copy(reference.CoupledField(), nativeCandidate);
            oracle.Commit();
            CHECK(driver.PreviousDt() == dt);
            time += dt;
        }
        if (method == "bdf2")
        {
            auto restored = driver.Fresh();
            restored->RestoreHistory(driver.PreviousState(), driver.PreviousDt());
            CHECK(restored->HasHistory());
            const auto before = state.ReadOwned();
            REQUIRE(restored->Prepare(state, candidate, time, .025, evaluate, solve));
            REQUIRE(oracle.Prepare(reference.CoupledField(), nativeCandidate, time, .025, nativeRate, nativeSolve));
            compare();
            if (fixture.rows)
                CHECK(before.Block("species")(0, 0) == state.ReadOwned().Block("species")(0, 0));
        }
    }
}
TEST_CASE("Equation algebra reserves every output before writing and time sources cannot be silently mutated")
{
    Fixture fixture;
    auto state = fixture.Build(Schema(5), 2, true);
    Fill(state);
    auto candidate = C::EquationDofOperations::NewVector(state);
    C::EquationDofOperations::Fill(candidate, -7);
    std::vector<uint64_t> versions;
    for (I iField = 0; iField < candidate.PhysicalFieldNumber(); ++iField)
        versions.push_back(candidate.PhysicalField(iField).Owned().Version());
    {
        const auto held = candidate.PhysicalField(2).Owned().Read();
        CHECK_THROWS(C::EquationDofOperations::Copy(candidate, state));
    }
    for (I iField = 0; iField < candidate.PhysicalFieldNumber(); ++iField)
    {
        CHECK(candidate.PhysicalField(iField).Owned().Version() == versions[iField]);
        const auto read = candidate.PhysicalField(iField).Owned().Read();
        const auto values = read.View<R::UniformTableRows>();
        for (I iRow = 0; iRow < fixture.rows; ++iRow)
            for (I j = 0; j < values.Width(); ++j)
                CHECK(values(iRow, j) == -7);
    }
    C::EquationTimeStepper driver(state, "bdf1");
    const auto invalid = [&](auto &, const auto &source, double, double, double, int)
    {
        auto write = const_cast<C::EquationDofs &>(source).WriteOwned();
        if (fixture.rows)
            write.Block("species")(0, 0) += 1;
        return true;
    };
    CHECK_THROWS(driver.Prepare(state, candidate, 0, .01, [](auto &, auto &, double, int) {}, invalid));
    CHECK_FALSE(driver.HasPendingTrial());
    CHECK_FALSE(driver.HasHistory());
    CheckValues(state);
}
int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    mpi.setWorld();
    doctest::Context context(argc, argv);
    const int result = context.run();
    MPI_Finalize();
    return result;
}
