/**
 * Standalone migration probe, compiled unchanged against legacy and native sources.
 * Records local mesh ordering and IEEE real bits, plus a first-order evaluator
 * pipeline and optional configured RunImplicitEuler advance. Matching probe bits
 * establish parity only for the selected build, ranks, threads and configuration;
 * neither the fixed-state pipeline nor a bounded advance claims convergence.
 *
 * Usage: probe <NS|NS_SA|NS_3D> <config|--defaults> <output-prefix-or-default-path>
 *              [repeat-count=5] [advance]
 */
#include "Euler/EulerSolver.hpp"
#if __has_include("Geom/Mesh/NativeMeshState.hpp")
#    include "Geom/Mesh/NativeMeshState.hpp"
#    define DNDS_MIGRATION_NATIVE 1
#else
#    define DNDS_MIGRATION_NATIVE 0
#endif
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    using Json = nlohmann::ordered_json;
    using DNDS::real;
    using Index = DNDS::index;
    using namespace DNDS::Euler;

    uint64_t Bits(real value)
    {
        static_assert(sizeof(real) == sizeof(uint64_t), "probe requires 64-bit real");
        uint64_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }

    template <class T>
    Json MatrixBits(const T &matrix)
    {
        Json result = Json::array();
        // Column-major logical order, independent of Eigen storage/alignment.
        for (Eigen::Index col = 0; col < matrix.cols(); ++col)
            for (Eigen::Index row = 0; row < matrix.rows(); ++row)
                result.push_back(Bits(matrix(row, col)));
        return result;
    }

    template <class F>
    double Timed(const DNDS::MPIInfo &mpi, F &&work)
    {
        MPI_Barrier(mpi.comm);
        const double start = MPI_Wtime();
        work();
        const double elapsed = MPI_Wtime() - start;
        double maximum = 0;
        MPI_Allreduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
        return maximum;
    }

    template <EulerModel model>
    void Probe(const DNDS::MPIInfo &mpi, const std::string &configPath,
               const std::string &prefix, int repeats, bool defaults, bool advance)
    {
        using Solver = EulerSolver<model>;
        using Eval = EulerEvaluator<model>;
        constexpr int width = Eval::nVarsFixed;
        const int variables = getNVars(model);
        Solver solver(mpi);
        if (defaults)
        {
            solver.ConfigureFromJson(prefix, false);
            return;
        }
        Json output;
        output["timings"]["setup_seconds"] = Timed(mpi, [&]
                                                   {
            solver.ConfigureFromJson(configPath, true);
            solver.ReadMeshAndInitialize(); });
        if (solver.getConfiguration().timeMarchControl.partitionMeshOnly)
        {
            output["partition_only"] = true;
            const auto path = prefix + ".rank" + std::to_string(mpi.rank) + ".json";
            std::ofstream stream(path);
            if (!stream)
                throw std::runtime_error("cannot open partition probe output: " + path);
            stream << output.dump(2) << '\n';
            if (!stream)
                throw std::runtime_error("cannot write partition probe output: " + path);
            return;
        }
        auto mesh = solver.getMesh();
        const auto audit = mesh->CurrentNativeState()->AuditOptions();
        output["mesh_audit"] = {{"persistent_identity", audit.persistentIdentity}, {"closure", audit.closure}};
        // Validate a trusted construction at a checkpoint outside its setup
        // timing; audit omission must not hide a changed physical mesh.
        output["timings"]["checkpoint_audit_seconds"] = Timed(mpi, [&]
                                                              { mesh->CurrentNativeState()->Audit(); });
        auto fv = solver.getVFV();
        auto &eval = solver.getEval();
        auto &u = solver.getU();
        auto &rec = solver.getURec();
        auto &recNew = solver.getURecNew();
        auto &source = solver.getJSource();
        auto &beta = solver.getBetaPP();
        auto &config = solver.getConfiguration();
        ArrayDOFV<width> rhs, increment, incrementNew;
        ArrayDOFV<1> dt, alpha;
        JacobianDiagBlock<width> diagonal;
        fv->BuildUDof(rhs, variables);
        fv->BuildUDof(increment, variables);
        fv->BuildUDof(incrementNew, variables);
        fv->BuildUDof(dt, 1);
        fv->BuildUDof(alpha, 1);
        eval.InitializeUDOF(u);
        u.trans.startPersistentPull();
        u.trans.waitPersistentPull();
        eval.FixUMaxFilter(u);
        u.trans.startPersistentPull();
        u.trans.waitPersistentPull();
        rec.setConstant(0.0);
        recNew.setConstant(0.0);
        std::vector<Eigen::VectorXd> initialU, initialBeta;
        for (Index cell = 0; cell < mesh->NumCellProc(); ++cell)
        {
            initialU.emplace_back(u[cell]);
            initialBeta.emplace_back(beta[cell]);
        }
        output["timings"]["repeats"] = Json::array();
        for (int repeat = 0; repeat < repeats; ++repeat)
        {
            for (Index cell = 0; cell < mesh->NumCellProc(); ++cell)
            {
                u[cell] = initialU[size_t(cell)];
                beta[cell] = initialBeta[size_t(cell)];
            }
            rec.setConstant(0.0);
            recNew.setConstant(0.0);
            alpha.setConstant(1.0);
            source.clearValues();
            diagonal.SetModeAndInit(config.eulerSettings.useScalarJacobian ? 0 : 1, variables, u);
            real minimumDt = 1e100;
            Json times;
            times["dt_seconds"] = Timed(mpi, [&]
                                        { eval.EvaluateDt(dt, u, rec, config.implicitCFLControl.CFL, minimumDt, 1e100,
                                                          config.implicitCFLControl.useLocalDt, 0.0); });
            times["rhs_seconds"] = Timed(mpi, [&]
                                         {
                eval.EvaluateRHS(rhs, source, u, rec, rec, beta, alpha, false, 0.0);
                rhs.trans.startPersistentPull();
                rhs.trans.waitPersistentPull(); });
            times["matrix_seconds"] = Timed(mpi, [&]
                                            { eval.LUSGSMatrixInit(diagonal, source, dt, 1e100, 1.0, u, rec, 0, 0.0); });
            increment.setConstant(0.0);
            incrementNew.setConstant(0.0);
            times["solve_seconds"] = Timed(mpi, [&]
                                           {
                eval.UpdateLUSGSForward(1.0, 0.0, rhs, u, increment, diagonal, incrementNew);
                eval.UpdateLUSGSBackward(1.0, 0.0, rhs, u, increment, diagonal, incrementNew);
                incrementNew.trans.startPersistentPull();
                incrementNew.trans.waitPersistentPull(); });
            increment = incrementNew;
            times["minimum_dt_bits"] = Bits(minimumDt);
            output["timings"]["repeats"].push_back(std::move(times));
        }

        // Scope the native lease to dumping; an optional advance may mutate mesh
        // or rebuild caches and must not inherit a retained read from this probe.
        {
#if DNDS_MIGRATION_NATIVE
            auto read = mesh->AcquireRead(mesh->CurrentNativeRequirements());
            auto view = read.LegacyHostView();
            const auto native = mesh->CurrentNativeState();
#else
            auto &view = *mesh;
#endif
            enum class Kind
            {
                Node,
                Cell,
                Boundary
            };
            auto original = [&](Kind kind, Index row) -> Index
            {
                const Index count = kind == Kind::Node ? view.NumNodeProc() : kind == Kind::Cell ? view.NumCellProc()
                                                                                                 : view.NumBndProc();
                if (row < 0 || row >= count)
                    return row; // Preserve absent and encoded-unavailable sentinels.
#if DNDS_MIGRATION_NATIVE
                const auto entity = kind == Kind::Node ? DNDS::Geom::EntityKind::Node : kind == Kind::Cell ? DNDS::Geom::EntityKind::Cell
                                                                                                           : DNDS::Geom::EntityKind::Bnd;
                return native->OriginalEntityId(entity, row);
#else
                if (kind == Kind::Node)
                    return mesh->node2nodeOrig(row, 0);
                if (kind == Kind::Cell)
                    return mesh->cell2cellOrig(row, 0);
                return mesh->bnd2bndOrig(row, 0);
#endif
            };
            auto ids = [&](const auto &row, Kind kind)
            {
                Json result = Json::array();
                for (DNDS::rowsize slot = 0; slot < row.size(); ++slot)
                    result.push_back(original(kind, row[slot]));
                return result;
            };
            auto periodicBits = [&](const auto &row)
            {
                Json result = Json::array();
                for (DNDS::rowsize slot = 0; slot < row.size(); ++slot)
                    result.push_back(uint8_t(row[slot]));
                return result;
            };
            output["periodic"] = view.isPeriodic;
            for (size_t group = 0; group < 4; ++group)
                output["periodic_frames"].push_back({{"translation_bits", MatrixBits(view.periodicInfo.translation[group].map())},
                                                     {"rotation_bits", MatrixBits(view.periodicInfo.rotation[group].map())},
                                                     {"center_bits", MatrixBits(view.periodicInfo.rotationCenter[group].map())}});
            output["counts"] = {{"nodes", view.NumNode()}, {"nodes_local", view.NumNodeProc()}, {"cells", view.NumCell()}, {"cells_local", view.NumCellProc()}, {"faces", view.NumFace()}, {"faces_local", view.NumFaceProc()}, {"bnds", view.NumBnd()}, {"bnds_local", view.NumBndProc()}};
            for (const auto *name : {"nodes", "cells", "faces", "bnds", "state"})
                output[name] = Json::array();
            for (Index node = 0; node < view.NumNodeProc(); ++node)
            {
                Json value{{"id", original(Kind::Node, node)}, {"owned", node < view.NumNode()}, {"xyz_bits", MatrixBits(view.coords[node])}};
                if (view.nodeWallDist.father)
                    value["wall_bits"] = MatrixBits(view.nodeWallDist[node]);
                output["nodes"].push_back(std::move(value));
            }
            for (Index cell = 0; cell < view.NumCellProc(); ++cell)
            {
                output["cells"].push_back({{"id", original(Kind::Cell, cell)}, {"owned", cell < view.NumCell()}, {"nodes", ids(view.cell2node[cell], Kind::Node)}, {"neighbors", ids(view.cell2cell[cell], Kind::Cell)}, {"elem", int(view.cellElemInfo(cell, 0).type)}, {"zone", view.GetCellZone(cell)}});
                if (view.isPeriodic)
                    output["cells"].back()["pbi"] = periodicBits(view.cell2nodePbi[cell]);
                if (cell >= view.NumCell())
                    continue;
                output["state"].push_back({{"id", original(Kind::Cell, cell)}, {"u_bits", MatrixBits(u[cell])}, {"rec_bits", MatrixBits(rec[cell])}, {"dt_bits", Bits(dt[cell](0))}, {"rhs_bits", MatrixBits(rhs[cell])}, {"inc_bits", MatrixBits(increment[cell])}, {"vol_bits", Bits(fv->GetCellVol(cell))}, {"bary_bits", MatrixBits(fv->GetCellBary(cell))}});
            }
            for (Index face = 0; face < view.NumFaceProc(); ++face)
            {
                output["faces"].push_back({{"owned", face < view.NumFace()},
                                           {"nodes", ids(view.face2node[face], Kind::Node)},
                                           {"cells", ids(view.face2cell[face], Kind::Cell)},
                                           {"zone", view.GetFaceZone(face)},
                                           {"area_bits", Bits(fv->GetFaceArea(face))},
                                           {"normal_bits", MatrixBits(fv->GetFaceNorm(face, -1))}});
                if (view.isPeriodic)
                    output["faces"].back()["pbi"] = periodicBits(view.face2nodePbi[face]);
            }
            for (Index boundary = 0; boundary < view.NumBndProc(); ++boundary)
            {
                output["bnds"].push_back({{"id", original(Kind::Boundary, boundary)}, {"owned", boundary < view.NumBnd()}, {"nodes", ids(view.bnd2node[boundary], Kind::Node)}, {"cells", ids(view.bnd2cell[boundary], Kind::Cell)}, {"zone", view.GetBndZone(boundary)}});
                if (view.isPeriodic)
                    output["bnds"].back()["pbi"] = periodicBits(view.bnd2nodePbi[boundary]);
            }
        }
        if (advance)
        {
            output["timings"]["advance_seconds"] = Timed(mpi, [&]
                                                         { solver.RunImplicitEuler(); });
            for (Index cell = 0; cell < mesh->NumCell(); ++cell)
            {
                output["state"][size_t(cell)]["final_u_bits"] = MatrixBits(u[cell]);
                output["state"][size_t(cell)]["final_rec_bits"] = MatrixBits(rec[cell]);
            }
        }
        const std::string path = prefix + ".rank" + std::to_string(mpi.rank) + ".json";
        std::ofstream stream(path);
        if (!stream)
            throw std::runtime_error("cannot open probe output: " + path);
        stream << output.dump(2) << '\n';
        if (!stream)
            throw std::runtime_error("cannot write probe output: " + path);
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    try
    {
        if (argc < 4 || argc > 6)
            throw std::invalid_argument("usage: probe <NS|NS_SA|NS_3D> <config|--defaults> <prefix-or-default-path> [positive-repeat-count=5] [advance]");
        const std::string model = argv[1];
        int repeats = 5;
        bool advance = false;
        if (argc >= 5)
        {
            if (std::string(argv[4]) == "advance")
                advance = true;
            else
            {
                size_t consumed = 0;
                const auto value = std::stoll(argv[4], &consumed);
                if (consumed != std::strlen(argv[4]) || value <= 0 || value > std::numeric_limits<int>::max())
                    throw std::invalid_argument("repeat count must be a positive integer");
                repeats = int(value);
            }
        }
        if (argc == 6)
        {
            if (advance || std::string(argv[5]) != "advance")
                throw std::invalid_argument("last argument must be advance");
            advance = true;
        }
        const bool defaults = std::string(argv[2]) == "--defaults";
        if (model == "NS")
            Probe<NS>(mpi, argv[2], argv[3], repeats, defaults, advance);
        else if (model == "NS_SA")
            Probe<NS_SA>(mpi, argv[2], argv[3], repeats, defaults, advance);
        else if (model == "NS_3D")
            Probe<NS_3D>(mpi, argv[2], argv[3], repeats, defaults, advance);
        else
            throw std::invalid_argument("model must be NS, NS_SA or NS_3D");
    }
    catch (const std::exception &error)
    {
        std::cerr << "migration probe rank " << mpi.rank << ": " << error.what() << std::endl;
        MPI_Abort(mpi.comm, 1);
        return 1;
    }
    MPI_Finalize();
    return 0;
}
