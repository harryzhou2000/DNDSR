// One numeric component per invocation: shared rank-local graph, ILU pattern,
// matrix and RHS. No mesh, physics, communication, symbolic setup or nonlinear
// convergence is timed. Both factorizations include resetting numeric storage.
#include "Euler/EulerJacobian.hpp"
#include "Solver/NativeBlockLU.hpp"
#include "Cauchy/Implicit/LinearBlockKernels.hpp"
#include <iostream>

namespace R = DNDS::Runtime;
namespace E = DNDS::Euler;
using I = DNDS::index;
constexpr int components = 6;

int main(int argc, char **argv)
{
    DNDS::MPI::Init_thread(&argc, &argv);
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    try
    {
        DNDS_check_throw_info(argc == 6, "arguments: factor|solve|logical-solve|logical-action local-grid-width fill repetitions trials");
        const std::string mode = argv[1];
        const I width = std::stoll(argv[2]);
        const int fill = std::stoi(argv[3]), repetitions = std::stoi(argv[4]), trials = std::stoi(argv[5]);
        DNDS_check_throw_info((mode == "factor" || mode == "solve" || mode == "logical-solve" || mode == "logical-action") && width >= 2 && width <= 128 &&
                                  fill >= 0 && fill <= 2 && repetitions > 0 && trials > 0,
                              "invalid local block comparison controls");
        DNDS_check_throw_info(mode != "logical-action" || fill == 0, "matrix action requires the same original no-fill graph on both sides");
        const I cells = width * width;
        std::vector<std::vector<I>> adjacency(cells);
        for (I iCell = 0; iCell < cells; ++iCell)
        {
            const I x = iCell % width, y = iCell / width;
            for (const auto other : {x ? iCell - 1 : I(-1), x + 1 < width ? iCell + 1 : I(-1),
                                     y ? iCell - width : I(-1), y + 1 < width ? iCell + width : I(-1)})
                if (other >= 0)
                    adjacency[iCell].push_back(other);
            std::sort(adjacency[iCell].begin(), adjacency[iCell].end());
        }
        auto symbolic = std::make_shared<DNDS::Direct::SerialSymLUStructure>(mpi, cells);
        symbolic->ObtainSymmetricSymbolicFactorization(adjacency, {0, cells}, fill);
        DNDS::Direct::NativeBlockLU native(symbolic, components);
        E::JacobianLocalLU<components> legacy(symbolic, components), original(symbolic, components);
        I blocks = 0;
        {
            auto write = native.Matrix().Write();
            const auto values = write.View();
            for (I iCell = 0; iCell < cells; ++iCell)
            {
                std::vector<I> columns{iCell};
                columns.insert(columns.end(), symbolic->lowerTriStructure[iCell].begin(), symbolic->lowerTriStructure[iCell].end());
                columns.insert(columns.end(), symbolic->upperTriStructure[iCell].begin(), symbolic->upperTriStructure[iCell].end());
                blocks += I(columns.size());
                for (I j = 0; j < I(columns.size()); ++j)
                {
                    Eigen::Matrix<double, components, components> block = Eigen::Matrix<double, components, components>::Zero();
                    const bool diagonal = j == 0;
                    const bool edge = std::binary_search(adjacency[iCell].begin(), adjacency[iCell].end(), columns[j]);
                    if (diagonal || edge)
                        for (int a = 0; a < components; ++a)
                            for (int b = 0; b < components; ++b)
                                block(a, b) = a == b ? (diagonal ? (a == 5 ? 32. : 8.) : -.7)
                                                     : (diagonal ? .02 : .003) * std::sin(double(1 + iCell + a * components + b));
                    values(iCell, j) = block;
                    original.LDU(iCell, j) = block;
                }
            }
        }
        const auto factorLegacy = [&]
        {
            legacy.isDecomposed = false;
            for (I iCell = 0; iCell < cells; ++iCell)
                for (I j = 0; j < 1 + I(symbolic->lowerTriStructure[iCell].size()) + I(symbolic->upperTriStructure[iCell].size()); ++j)
                    legacy.LDU(iCell, j) = original.LDU(iCell, j);
            legacy.InPlaceDecompose();
        };
        R::Array<double> rhs(R::RowLayout::Dense(cells, components)), forward(rhs.Layout()), output(rhs.Layout());
        E::ArrayDOFV<components> legacyRHS, legacyOutput;
        for (auto *field : {&legacyRHS, &legacyOutput})
        {
            field->InitPair("block-benchmark", mpi);
            field->father->Resize(cells, components, 1);
            field->son->Resize(0, components, 1);
        }
        {
            auto write = rhs.Write();
            const auto values = write.View<R::FixedRows<components>>();
            for (I iCell = 0; iCell < cells; ++iCell)
                for (int j = 0; j < components; ++j)
                    values(iCell, j) = legacyRHS[iCell][j] = std::sin(.03 * (iCell + j + 1));
        }
        const auto rhsVersion = rhs.Version();
        native.Factorize();
        factorLegacy();
        struct ActionPattern
        {
            struct Partition
            {
                I owned;
            } partitionStorage;
            const Partition *partition = &partitionStorage;
            const std::vector<std::vector<I>> &columns;
        } pattern{{cells}, nullptr, adjacency};
        pattern.partition = &pattern.partitionStorage;
        std::vector<I> rows(cells);
        std::iota(rows.begin(), rows.end(), I(0));
        const auto solveNative = [&]
        {
            if (mode != "logical-solve" && mode != "logical-action")
            {
                native.Solve(rhs, forward, output);
                return;
            }
            const auto read = rhs.Read();
            auto middle = forward.Write(), write = output.Write();
            std::vector<DNDS::Algebra::DofComponentColumn<const double>> inputColumns;
            std::vector<DNDS::Algebra::DofComponentColumn<double>> middleColumns, outputColumns;
            inputColumns.reserve(components);
            middleColumns.reserve(components);
            outputColumns.reserve(components);
            for (int j = 0; j < components; ++j)
            {
                inputColumns.push_back({read.View<R::UniformTableRows>(), {}, j});
                middleColumns.push_back({middle.View<R::UniformTableRows>(), {}, j});
                outputColumns.push_back({write.View<R::UniformTableRows>(), {}, j});
            }
            const DNDS::Algebra::DofComponentLayout<const double> b(std::move(inputColumns), cells, 0);
            const DNDS::Algebra::DofComponentLayout<double> y(std::move(middleColumns), cells, 0), x(std::move(outputColumns), cells, 0);
            if (mode == "logical-action")
            {
                const auto matrix = native.Matrix().Read();
                struct Diagonal
                {
                    DNDS::Algebra::BlockMatrixView<const double> values;
                    auto operator[](I iCell) const { return values(iCell, 0); }
                } diagonal{matrix.View()};
                const auto blocks = [values = matrix.View()](I iCell, I j)
                { return values(iCell, 1 + j); };
                DNDS::Cauchy::Detail::ApplyBlockRows(b.View(), b.View().GhostPart(), x.View(), diagonal, blocks, pattern, rows, components, true);
            }
            else
                native.SolveRows(b.View(), y.View(), x.View());
        };
        const auto solveLegacy = [&]
        {
            if (mode == "logical-action")
                original.MatMul(legacyRHS, legacyOutput);
            else
                legacy.Solve(legacyRHS, legacyOutput);
        };
        const auto check = [&]
        {
            solveNative();
            solveLegacy();
            const auto read = output.Read();
            const auto values = read.View<R::FixedRows<components>>();
            double error = 0;
            for (I iCell = 0; iCell < cells; ++iCell)
                for (int j = 0; j < components; ++j)
                {
                    const double difference = std::abs(values(iCell, j) - legacyOutput[iCell][j]);
                    DNDS_check_throw_info(std::isfinite(difference), "nonfinite local factor solve");
                    error = std::max(error, difference);
                }
            MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
            DNDS_check_throw_info(error < 1e-12 && native.LastInverseFallbacks() == 0 && rhs.Version() == rhsVersion,
                                  "local block solve mismatch or unexpected inverse fallback/input mutation");
            return error;
        };
        double error = check();
        const auto measure = [&](const auto &run)
        {
            MPI_Barrier(mpi.comm);
            const double start = MPI_Wtime();
            for (int repeat = 0; repeat < repetitions; ++repeat)
                run();
            const double elapsed = (MPI_Wtime() - start) / repetitions;
            double maximum = 0;
            MPI_Allreduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
            return maximum;
        };
        const auto runNative = [&]
        { if (mode == "factor") native.Factorize(); else solveNative(); };
        const auto runLegacy = [&]
        { if (mode == "factor") factorLegacy(); else solveLegacy(); };
        for (int trial = 0; trial < trials; ++trial)
        {
            double nativeSeconds, legacySeconds;
            if (trial % 2)
                legacySeconds = measure(runLegacy), nativeSeconds = measure(runNative);
            else
                nativeSeconds = measure(runNative), legacySeconds = measure(runLegacy);
            error = check();
            if (!mpi.rank)
                std::cout << "BLOCK_LU_COMPONENT " << nlohmann::ordered_json{{"mode", mode}, {"local_cells", cells}, {"components", components}, {"np", mpi.size}, {"fill", fill}, {"local_factor_blocks", blocks}, {"repetitions", repetitions}, {"trial", trial}, {"native_seconds", nativeSeconds}, {"legacy_seconds", legacySeconds}, {"max_solution_error", error}, {"max_output_error", error}, {"includes_matrix_reset", mode == "factor"}, {"includes_mpi", false}, {"rhs_generation_preserved", true}}.dump() << std::endl;
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << "Block LU benchmark rank " << mpi.rank << ": " << error.what() << std::endl;
        MPI_Abort(mpi.comm, 1);
    }
    DNDS::MPI::Finalize();
}
