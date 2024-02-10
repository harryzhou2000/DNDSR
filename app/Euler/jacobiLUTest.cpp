#include "Euler/EulerSolver.hpp"

static DNDS::MPIInfo mpi;
static const int bDim = 5;

namespace DNDS::Euler
{
    void jacobiLUTest()
    {
        DNDS_assert(mpi.size == 1);
        ssp<Geom::UnstructuredMesh> mesh;
        DNDS_MAKE_SSP(mesh, mpi, 2);
        auto reader = DNDS::Geom::UnstructuredMeshSerialRW(mesh, 0);
        mesh->periodicInfo.translation[1] *= 10;
        mesh->periodicInfo.translation[2] *= 10;
        reader.ReadFromCGNSSerial("../data/mesh/IV10_160.cgns");
        reader.Deduplicate1to1Periodic();
        reader.BuildCell2Cell();
        reader.MeshPartitionCell2Cell();
        reader.PartitionReorderToMeshCell2Cell();
        reader.BuildSerialOut();
        mesh->BuildGhostPrimary();
        mesh->AdjGlobal2LocalPrimary();
        mesh->InterpolateFace(); // this mesh building is copied from meshSerial_Test

        mesh->ObtainLocalFactFillOrdering(0);
        mesh->ObtainSymmetricSymbolicFactorization(5);

        JacobianLocalLU<bDim> J(mesh, bDim);
        J.setZero();
        auto cell2cellFaceV = mesh->GetCell2CellFaceVLocal();
        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            index iCellP = mesh->CellFillingReorderOld2New(iCell);
            J.GetDiag(iCell).setIdentity();
            J.GetDiag(iCell) *= 0.5;
            J.GetDiag(iCell) += decltype(J.GetDiag(iCell))::Constant(0.5);
            for (auto iCO : cell2cellFaceV[iCell])
            {
                index iCOP = mesh->CellFillingReorderOld2New(iCO);
                if (iCOP < iCellP)
                {
                    auto ret = std::lower_bound(
                        mesh->lowerTriStructureNew[iCellP].begin(),
                        mesh->lowerTriStructureNew[iCellP].end(), iCOP);
                    DNDS_assert(ret != mesh->lowerTriStructureNew[iCellP].end());
                    J.GetLower(iCell, ret - mesh->lowerTriStructureNew[iCellP].begin()).setConstant(.1);
                }
                if (iCOP > iCellP)
                {
                    auto ret = std::lower_bound(
                        mesh->upperTriStructureNew[iCellP].begin(),
                        mesh->upperTriStructureNew[iCellP].end(), iCOP);
                    DNDS_assert(ret != mesh->upperTriStructureNew[iCellP].end());
                    J.GetUpper(iCell, ret - mesh->upperTriStructureNew[iCellP].begin()).setConstant(-.1);
                }
            }
        }

        ArrayDOFV<bDim> solAcc, sol, b;
        DNDS_MAKE_SSP(solAcc.father, mpi);
        DNDS_MAKE_SSP(solAcc.son, mpi);
        DNDS_MAKE_SSP(sol.father, mpi);
        DNDS_MAKE_SSP(sol.son, mpi);
        DNDS_MAKE_SSP(b.father, mpi);
        DNDS_MAKE_SSP(b.son, mpi);

        solAcc.father->Resize(mesh->NumCell(), bDim, 1);
        sol.father->Resize(mesh->NumCell(), bDim, 1);
        b.father->Resize(mesh->NumCell(), bDim, 1);
        // solAcc.setConstant(1.0);
        for (index i = 0; i < b.Size(); i++)
            solAcc[i].setConstant(real(i) / real(mesh->NumCell()));
        // solAcc[i].setConstant(1);
        J.MatMul(solAcc, b);
        // for (index i = 0; i < b.Size(); i++)
        //     std::cout << b[i].transpose() << ", ";
        // std::cout << std::endl;

        double t0 = MPI_Wtime();
        J.InPlaceDecompose();
        std::cout << "Decomposing finished for " << std::scientific << (MPI_Wtime() - t0) << std::endl;
        // J.PrintLog();
        t0 = MPI_Wtime();
        J.Solve(b, sol);
        std::cout << "Solving finished for " << std::scientific << (MPI_Wtime() - t0) << std::endl;
        std::cout << "solved" << std::endl;
        solAcc -= sol;
        std::cout << "Error is " << std::scientific << std::setprecision(10) << solAcc.norm2() << std::endl;

        // for (index i = 0; i < b.Size(); i++)
        //     std::cout << sol[i].transpose() << ", ";
        // std::cout << std::endl;
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    mpi.setWorld();
    DNDS::Euler::jacobiLUTest();
    MPI_Finalize();
}
