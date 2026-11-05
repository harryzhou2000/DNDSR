// Compile this unchanged against each pinned Euler source. No time integration.
// The optional seed prefix imports exact conservative bits from the old run,
// keyed by persistent cell ID, so conversion differences cannot alter RHS input.
#include "Euler/EulerSolver.hpp"
#include <fstream>
#include <cstring>
#include <map>

#if __has_include("Geom/Unstructured/UnstructuredMeshState.hpp")
#    define DNDS_FROZEN_NATIVE_MESH 1
#else
#    define DNDS_FROZEN_NATIVE_MESH 0
#endif

namespace
{
    using J = nlohmann::ordered_json;
    using I = DNDS::index;
    using DNDS::real;
    using namespace DNDS::Euler;
    uint64_t Bits(real value)
    {
        uint64_t result;
        static_assert(sizeof(result) == sizeof(value));
        std::memcpy(&result, &value, sizeof(value));
        return result;
    }
    real FromBits(uint64_t bits)
    {
        real result;
        std::memcpy(&result, &bits, sizeof(result));
        return result;
    }
    template <class Matrix>
    J MatrixBits(const Matrix &values)
    {
        J result = J::array();
        for (Eigen::Index column = 0; column < values.cols(); ++column)
            for (Eigen::Index row = 0; row < values.rows(); ++row)
            {
                DNDS_check_throw_info(std::isfinite(values(row, column)), "nonfinite frozen-probe output");
                result.push_back(Bits(values(row, column)));
            }
        return result;
    }
    template <EulerModel model>
    void Probe(const DNDS::MPIInfo &mpi, const std::string &configPath, const std::string &prefix,
               int polynomial, const std::string &seedPrefix)
    {
        using Solver = EulerSolver<model>;
        using Eval = EulerEvaluator<model>;
        constexpr int width = Eval::nVarsFixed;
        const int variables = model == NS_EX ? 14 : 6;
        Solver solver(mpi, variables);
        // ConfigureFromJson reads a complete document; the application normally
        // supplies generated defaults before merging a hand-maintained patch.
        const std::string defaults = prefix + ".default.json";
        solver.ConfigureFromJson(defaults, false);
        MPI_Barrier(mpi.comm);
        solver.ConfigureFromJson(defaults, true, configPath);
        solver.ReadMeshAndInitialize();
        auto mesh = solver.getMesh();
        auto fv = solver.getVFV();
        auto &eval = solver.getEval();
        auto &u = solver.getU();
        auto &rec = solver.getURec();
        auto &source = solver.getJSource();
        auto &beta = solver.getBetaPP();
        ArrayDOFV<width> rhs, withoutReactive, withoutJacobian, withoutSources;
        ArrayDOFV<1> dt, alpha;
        for (auto *field : {&rhs, &withoutReactive, &withoutJacobian, &withoutSources})
            fv->BuildUDof(*field, variables);
        fv->BuildUDof(dt, 1);
        fv->BuildUDof(alpha, 1);
        alpha.setConstant(1);
        beta.setConstant(1);
        source.SetModeAndInit(1, variables, u);
        std::map<I, J> seeds;
        if (seedPrefix != "-")
            for (int rank = 0; rank < mpi.size; ++rank)
            {
                std::ifstream stream(seedPrefix + ".rank" + std::to_string(rank) + ".json");
                DNDS_check_throw_info(bool(stream), "missing old-source frozen seed record");
                J record;
                stream >> record;
                DNDS_check_throw_info(record.at("model") == int(model) && record.at("polynomial") == polynomial,
                                      "frozen seed model/polynomial differs");
                for (const auto &cell : record.at("cells"))
                    DNDS_check_throw_info(seeds.emplace(cell.at("id").get<I>(), cell.at("u_bits")).second,
                                          "duplicate persistent ID in frozen seed");
            }
        const auto original = [&](I iCell)
        {
#if DNDS_FROZEN_NATIVE_MESH
            return mesh->OriginalEntityId(DNDS::Geom::EntityKind::Cell, iCell);
#else
            return mesh->cell2cellOrig(iCell, 0);
#endif
        };
        eval.InitializeUDOF(u);
        std::vector<J> converted;
        for (I iCell = 0; iCell < mesh->NumCell(); ++iCell)
        {
            const auto point = fv->GetCellBary(iCell);
            const real wave = std::sin(.37 * point[0]) * std::cos(.23 * point[1]);
            typename Solver::TU primitive, conservative;
            primitive.resize(variables);
            primitive.setZero();
            primitive[0] = 1.1 + .03 * wave;
            primitive[1] = .2 + .02 * std::cos(.21 * point[0]);
            primitive[2] = -.1 + .01 * std::sin(.19 * point[1]);
            primitive[3] = .05;
            primitive[4] = (model == NS_EX ? 1400 + 40 * wave : 300 + 5 * wave) / eval.phys().T0();
            if constexpr (model == NS_EX)
            {
                DNDS_check_throw_info(eval.phys().hasChemicalSource() && eval.phys().nSpecies() == 10,
                                      "reacting probe requires the ten-species h2o2 material");
                for (int k = 0; k < 9; ++k)
                    primitive[5 + k] = 1e-6;
                primitive[5] = .028 + .0002 * wave;
                primitive[8] = .22 - .0001 * wave;
                primitive[10] = .02;
            }
            else
                primitive[5] = eval.phys().muRef() / primitive[0] * (original(iCell) % 3 == 0 ? -.2 : 5.);
            eval.phys().primRhoTToConservative(primitive, conservative);
            converted.push_back(MatrixBits(conservative));
            if (seedPrefix != "-")
            {
                const auto &values = seeds.at(original(iCell));
                DNDS_check_throw_info(values.size() == size_t(variables), "frozen state width differs");
                for (int component = 0; component < variables; ++component)
                    conservative[component] = FromBits(values[component].template get<uint64_t>());
            }
            u[iCell] = conservative;
            auto values = rec[iCell];
            for (Eigen::Index mode = 0; mode < values.rows(); ++mode)
                for (int component = 0; component < variables; ++component)
                    values(mode, component) = polynomial ? 1e-5 * (std::abs(conservative[component]) + 1e-10) *
                                                               std::sin(.17 * original(iCell) + .31 * mode + .13 * component)
                                                         : 0;
        }
        u.trans.startPersistentPull();
        u.trans.waitPersistentPull();
        rec.trans.startPersistentPull();
        rec.trans.waitPersistentPull();
        std::vector<J> frozenU, frozenRec;
        for (I iCell = 0; iCell < mesh->NumCellProc(); ++iCell)
        {
            frozenU.push_back(MatrixBits(u[iCell]));
            frozenRec.push_back(MatrixBits(rec[iCell]));
        }
        real minimumDt = 1e100;
        eval.EvaluateDt(dt, u, rec, 1., minimumDt, 1e100, false, 0.);
        constexpr uint64_t quiet = Eval::RHS_Dont_Update_Integration | Eval::RHS_Dont_Record_Bud_Flux;
        eval.settings.ignoreSourceTerm = false;
        eval.EvaluateRHS(rhs, source, u, rec, rec, beta, alpha, false, 0., quiet);
        std::vector<J> jacobian;
        for (I iCell = 0; iCell < mesh->NumCell(); ++iCell)
            jacobian.push_back(MatrixBits(source.getValue(iCell)));
        eval.EvaluateRHS(withoutReactive, source, u, rec, rec, beta, alpha, false, 0., quiet | Eval::RHS_Ignore_Reactive_Source);
        eval.EvaluateRHS(withoutJacobian, source, u, rec, rec, beta, alpha, false, 0., quiet | Eval::RHS_Ignore_Reactive_Source_Jacobian);
        std::vector<J> noReactiveJacobian;
        for (I iCell = 0; iCell < mesh->NumCell(); ++iCell)
            noReactiveJacobian.push_back(MatrixBits(source.getValue(iCell)));
        eval.settings.ignoreSourceTerm = true;
        eval.EvaluateRHS(withoutSources, source, u, rec, rec, beta, alpha, false, 0., quiet);
        eval.settings.ignoreSourceTerm = false;
        std::optional<ArrayDOFV<width>> repaired;
        if constexpr (model == NS_EX)
        {
            repaired.emplace();
            fv->BuildUDof(*repaired, variables);
            for (I iCell = 0; iCell < mesh->NumCell(); ++iCell)
            {
                (*repaired)[iCell] = u[iCell];
                (*repaired)[iCell](6) = -u[iCell](0) * 1e-12;
            }
            eval.RepairCellMeanState(*repaired);
        }
        J cells = J::array();
        for (I iCell = 0; iCell < mesh->NumCell(); ++iCell)
        {
            typename Solver::TU state = u[iCell], primitive, cellSource;
            typename Solver::TDiffU gradient;
            typename Solver::TJacobianU cellJacobian;
            gradient.resize(Eval::dim, variables);
            gradient.setZero();
            gradient(0, 2) = .01;
            cellSource.resize(variables);
            cellSource.setZero();
            eval.EvaluateCellSource(cellSource, cellJacobian, state, gradient, iCell, 2);
            eval.phys().conservativeToPrimitive(state, primitive);
            const real temperature = eval.phys().temperature(state), pressure = primitive[4];
            Eigen::VectorXd properties(6);
            properties << temperature, pressure, eval.phys().Cp(temperature, state), eval.phys().Cv(temperature, state),
                eval.phys().mixtureViscosity(temperature, pressure, state), eval.phys().mixtureConductivity(temperature, pressure, state);
            J row{{"id", original(iCell)}, {"u_bits", frozenU[iCell]}, {"converted_u_bits", converted[iCell]}, {"rec_bits", frozenRec[iCell]}, {"rhs_bits", MatrixBits(rhs[iCell])}, {"rhs_no_reactive_bits", MatrixBits(withoutReactive[iCell])}, {"rhs_no_reactive_jacobian_bits", MatrixBits(withoutJacobian[iCell])}, {"rhs_no_sources_bits", MatrixBits(withoutSources[iCell])}, {"source_jacobian_bits", jacobian[iCell]}, {"no_reactive_jacobian_bits", noReactiveJacobian[iCell]}, {"cell_source_bits", MatrixBits(cellSource)}, {"cell_jacobian_bits", MatrixBits(cellJacobian)}, {"properties_bits", MatrixBits(properties)}, {"volume_bits", Bits(fv->GetCellVol(iCell))}, {"dt_bits", Bits(dt[iCell](0))}};
            if (eval.phys().hasChemicalSource())
            {
                Eigen::VectorXd diffusion(eval.phys().nSpecies());
                eval.phys().mixtureDiffusivity(temperature, pressure, state, diffusion);
                row["diffusion_bits"] = MatrixBits(diffusion);
                row["species_repair_bits"] = MatrixBits((*repaired)[iCell]);
            }
            cells.push_back(std::move(row));
        }
        for (I iCell = 0; iCell < mesh->NumCellProc(); ++iCell)
            DNDS_check_throw_info(MatrixBits(u[iCell]) == frozenU[iCell] && MatrixBits(rec[iCell]) == frozenRec[iCell],
                                  "Euler probe changed its frozen inputs");
        const auto &gas = eval.settings.idealGasProperty;
        J output{{"model", int(model)}, {"np", mpi.size}, {"rank", mpi.rank}, {"polynomial", polynomial}, {"seed_prefix", seedPrefix}, {"energy_convention", "total internal energy including species reference energy"}, {"scales", {gas.rho0, gas.U0, gas.L0, gas.T0}}, {"input_bits_preserved", true}, {"cells", cells}};
        std::ofstream stream(prefix + ".rank" + std::to_string(mpi.rank) + ".json");
        DNDS_check_throw_info(bool(stream), "cannot create frozen Euler evidence");
        stream << output.dump() << '\n';
        DNDS_check_throw_info(bool(stream), "cannot write frozen Euler evidence");
    }
}
int main(int argc, char **argv)
{
    const int initialized = DNDS::MPI::Init_thread(&argc, &argv);
    if (initialized != MPI_SUCCESS)
        return initialized;
    try
    {
        DNDS_check_throw_info(argc == 6, "arguments: NS_EX|NS_SA config output-prefix polynomial(0|1) old-seed-prefix-or-dash");
        const int polynomial = std::stoi(argv[4]);
        DNDS_check_throw_info(polynomial == 0 || polynomial == 1, "frozen polynomial mode must be zero or one");
        const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
        if (std::string(argv[1]) == "NS_EX")
            Probe<DNDS::Euler::NS_EX>(mpi, argv[2], argv[3], polynomial, argv[5]);
        else
        {
            DNDS_check_throw_info(std::string(argv[1]) == "NS_SA", "unknown frozen Euler model");
            Probe<DNDS::Euler::NS_SA>(mpi, argv[2], argv[3], polynomial, argv[5]);
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    DNDS::MPI::Finalize();
    return 0;
}
