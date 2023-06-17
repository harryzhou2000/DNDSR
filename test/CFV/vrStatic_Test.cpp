#include "CFV/VariationalReconstruction.hpp"
#include "Solver/Linear.hpp"
#include "Euler/Euler.hpp"

#include <cstdlib>
#include <omp.h>

std::vector<double> argD;

/*
running:
valgrind --log-file=log_valgrind.log
*/
struct TEvalDub
{
    static const int nVars_Fixed = 1;
};
void staticReconstruction()
{
    static const int dim = 2;

    auto mpi = DNDS::MPIInfo();
    mpi.setWorld();
    // DNDS::Debug::MPIDebugHold(mpi);
    // DNDS::Debug::MPIDebugHold(mpi);
    char buf[512];
    // std::cout << getcwd(buf, 512) << std::endl;
    auto mesh = std::make_shared<DNDS::Geom::UnstructuredMesh>(mpi, dim);
    auto reader = DNDS::Geom::UnstructuredMeshSerialRW(mesh, 0);
    // "../data/mesh/FourTris_V1.pw.cgns"
    // "../data/mesh/SC20714_MixedA.cgns"
    // "../data/mesh/UniformDM240_E120.cgns"
    // "../data/mesh/Ball.cgns"
    // "../data/mesh/UniformSquare_10.cgns"
    reader.ReadFromCGNSSerial("../data/mesh/UniformSquare_80.cgns");
    reader.Deduplicate1to1Periodic();

    reader.BuildCell2Cell();
    reader.MeshPartitionCell2Cell();
    reader.PartitionReorderToMeshCell2Cell();
    reader.BuildSerialOut();
    mesh->BuildGhostPrimary();
    mesh->AdjGlobal2LocalPrimary();
    mesh->InterpolateFace();
    mesh->AssertOnFaces();

    auto vr = DNDS::CFV::VariationalReconstruction<dim>(mpi, mesh);
#ifdef DNDS_USE_OMP
    omp_set_num_threads(DNDS::MPIWorldSize() == 1 ? std::min(omp_get_num_procs(), omp_get_max_threads()) : 1);
#endif
    // omp_set_num_threads(1);

    vr.settings.maxOrder = 3;
    vr.settings.SORInstead = true;
    vr.settings.cacheDiffBase = true;

    vr.settings.smoothThreshold = 1e100;

    vr.ConstructMetrics();
    vr.ConstructBaseAndWeight();
    vr.ConstructRecCoeff();

    {
        using namespace DNDS;
        using namespace DNDS::Geom;
        using namespace DNDS::CFV;

        auto fScalar = [](const tPoint &p)
        {
            real kdpi = 5;
            return Eigen::Vector<real, 4>{
                std::cos(pi * kdpi * p[0]) * std::cos(pi * kdpi * p[1]) * std::cos(pi * kdpi * p[2]),
                -pi * kdpi * std::sin(pi * kdpi * p[0]) * std::cos(pi * kdpi * p[1]) * std::cos(pi * kdpi * p[2]),
                -pi * kdpi * std::sin(pi * kdpi * p[1]) * std::cos(pi * kdpi * p[2]) * std::cos(pi * kdpi * p[0]),
                -pi * kdpi * std::sin(pi * kdpi * p[2]) * std::cos(pi * kdpi * p[0]) * std::cos(pi * kdpi * p[1]),
            };
        };

        tUDof<1> u;
        vr.BuildUDof(u, 1);
        for (DNDS::index iCell = 0; iCell < vr.mesh->NumCell(); iCell++)
        {
            auto qCell = vr.GetCellQuad(iCell);
            Eigen::Vector<real, 1> uc;
            uc.setZero();
            qCell.IntegrationSimple(
                uc,
                [&](auto &vInc, int iG)
                {
                    vInc = fScalar(vr.GetCellQuadraturePPhys(iCell, iG))({0}) * vr.GetCellJacobiDet(iCell, iG);
                });
            u[iCell] = uc / vr.GetCellVol(iCell);
            // std::cout << iCell << " " << u[iCell].transpose() << std::endl;
        }
        u.trans.startPersistentPull();
        u.trans.waitPersistentPull();

        ssp<Euler::ArrayRECV<1>> uRec, uRecNew, uRecOld, uRecCur;
        uRec = std::make_shared<Euler::ArrayRECV<1>>();
        uRecNew = std::make_shared<Euler::ArrayRECV<1>>();
        uRecOld = std::make_shared<Euler::ArrayRECV<1>>();
        uRecCur = std::make_shared<Euler::ArrayRECV<1>>();

        vr.BuildURec(*uRec, 1);
        vr.BuildURec(*uRecNew, 1);
        vr.BuildURec(*uRecOld, 1);
        vr.BuildURec(*uRecCur, 1);

        auto printErr = [&](Euler::ArrayRECV<1> &uRecC)
        {
            Eigen::Vector<real, 4> err, errAll;
            err.setZero();
            for (DNDS::index iCell = 0; iCell < vr.mesh->NumCell(); iCell++)
            {
                auto qCell = vr.GetCellQuad(iCell);
                decltype(err) errC;
                errC.setZero();
                qCell.IntegrationSimple(
                    errC,
                    [&](auto &vInc, int iG)
                    {
                        Eigen::VectorXd udu =
                            (vr.GetIntPointDiffBaseValue(iCell, -1, -1, iG, std::array<int, 4>{0, 1, 2, 3}, 4) * uRecC[iCell]);
                        // std::cout << udu.transpose() << std::endl;
                        udu(0) += u[iCell](0);
                        vInc = (udu - fScalar(vr.GetCellQuadraturePPhys(iCell, iG))).array().abs() * vr.GetCellJacobiDet(iCell, iG);
                    });
                err += errC;
                // std::cout << si(iCell, 0) << std::endl;
            }
            MPI_Allreduce(err.data(), errAll.data(), 4, DNDS_MPI_REAL, MPI_SUM, mpi.comm);
            if (mpi.rank == 0)
            {
                // std::cout << "Err: [";
                std::cout << errAll.transpose() / vr.GetGlobalVol() << std::endl;
            }
        };

        /*************************************/
        // Do reconstruction iteration
        /*************************************/
        int kGmres = 5;
        int method = 1;
        std::cout << std::setprecision(16) << std::endl;

        for (int iter = 1; iter <= (method == 0 ? 100 : 0); iter++)
        {
            uRecOld->CopyFather(*uRec);
            vr.DoReconstructionIter(
                *uRec, *uRecNew, u,
                [&](const auto &uL, const auto &uMean, const tPoint &unitNorm, const tPoint &p, t_index faceID)
                {
                    // std::cout << p.transpose() << " do bnd" << std::endl;
                    return Eigen::Vector<real, 1>(fScalar(p)({0}));
                },
                true);

            real duRecSum = 0;
            for (DNDS::index iCell = 0; iCell < vr.mesh->NumCell(); iCell++)
                duRecSum += ((*uRecNew)[iCell] - (*uRecOld)[iCell]).array().square().sum();
            real duRecSumAll = 0;
            MPI_Allreduce(&duRecSum, &duRecSumAll, 1, DNDS_MPI_REAL, MPI_SUM, vr.mpi.comm);
            duRecSumAll /= vr.mesh->NumCellGlobal();

            std::swap(uRec, uRecNew);
            uRec->trans.startPersistentPull();
            uRec->trans.waitPersistentPull();
            // std::cout << (*uRec)[0].transpose() << std::endl;
            if (mpi.rank == 0)
            {
                std::cout << std::sqrt(duRecSumAll) << std::endl;
            }
            // printErr(*uRec);
        }

        
        Linear::GMRES_LeftPreconditioned<std::remove_reference_t<decltype(*uRec)>> gmresRec(
            kGmres,
            [&](decltype(*uRec) &data)
            {
                vr.BuildURec(data, 1);
            });
        {
            for (int iRec = 1; iRec <= (method == 1 ? 1: 0); iRec++)
            {
                if (mpi.rank == 0)
                    std::cout << "GMRESOut: " << std::endl;
                (*uRecOld) = (*uRec);
                vr.DoReconstructionIter(
                    *uRec, *uRecNew, u,
                    // FBoundary
                    [&](const auto &uL, const auto &uMean, const tPoint &unitNorm, const tPoint &p, t_index faceID)
                    {
                        // std::cout << p.transpose() << " do bnd" << std::endl;
                        return Eigen::Vector<real, 1>(fScalar(p)({0}));
                    },
                    true, true);
                uRecNew->trans.startPersistentPull();
                uRecNew->trans.waitPersistentPull();
                *uRec = *uRecNew;
                auto FBoundaryDiff =
                    [&](const auto &uL, const auto &duL, const auto &uMean, const tPoint &unitNorm, const tPoint &p, t_index faceID)
                {
                    return Eigen::Vector<real, 1>::Ones() * 0;
                };
                gmresRec.solve(
                    [&](decltype(*uRec) &x, decltype(*uRec) &Ax)
                    {
                        vr.DoReconstructionIterDiff(*uRecOld, x, Ax, u, FBoundaryDiff);
                        Ax.trans.startPersistentPull();
                        Ax.trans.waitPersistentPull();
                    },
                    [&](decltype(*uRec) &x, decltype(*uRec) &MLx)
                    {
                        MLx = x;
                        // MLx.trans.startPersistentPull();
                        // MLx.trans.waitPersistentPull();
                        // vr.DoReconstructionIterSOR(*uRecOld, x, MLx, u, FBoundaryDiff);
                    },
                    *uRecNew, *uRec, 100 / kGmres,
                    [&](uint32_t i, real res, real resB) -> bool
                    {
                        if (i >= 0)
                        {
                            if (mpi.rank == 0)
                            {
                                log() << std::scientific;
                                log() << res << std::endl;
                                *uRecCur = *uRecOld;
                                *uRecCur -= *uRec;
                                // printErr(*uRecCur);
                            }
                        }
                        return false;
                    });
                for (DNDS::index iCell = 0; iCell < uRec->Size(); iCell++)
                {
                    (*uRec)[iCell] = (*uRecOld)[iCell] - (*uRec)[iCell];
                }
                uRec->trans.startPersistentPull();
                uRec->trans.waitPersistentPull();
            }
        }

        tScalarPair si;
        vr.BuildScalar(si);
        vr.DoCalculateSmoothIndicator(si, *uRec, u, std::array<int, 1>{0});

        vr.DoLimiterWBAP_C(
            TEvalDub{},
            u, *uRec, *uRecNew, *uRecOld, si, false,
            [](const auto &uL, const auto &uR, const auto &uNorm)
            { return 1; },
            [](const auto &uL, const auto &uR, const auto &uNorm)
            { return 1; });
        printErr(*uRec);
    }
}

int main(int argc, char *argv[])
{
    // ! Disable MPI call to help serial mem check
    MPI_Init(&argc, &argv);

    for (int i = 1; i < argc; i++)
    {
        double v = std::atof(argv[i]);
        argD.push_back(v);
    }

    staticReconstruction();

    MPI_Finalize();

    return 0;
}