
#include "VariationalReconstruction.hpp"

namespace DNDS::CFV
{
    template <int dim>
    void
    VariationalReconstruction<dim>::
        ConstructMetrics()
    {
        using namespace Geom;
        using namespace Geom::Elem;

        /***************************************/
        // get volumes
        real sumVolume{0};
        volumeLocal.resize(
            mesh->NumCellProc());
        for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++)
        {
            auto eCell = mesh->GetCellElement(iCell);
            auto qCell = Quadrature{eCell, INT_ORDER_MAX};
            real v{0};
            tSmallCoords coordsCell;
            mesh->GetCoords(mesh->cell2node[iCell], coordsCell);

            qCell.Integration(
                v,
                [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                {
                    tJacobi J = Elem::ShapeJacobianCoordD01Nj(coordsCell, DiNj);
                    real JDet;
                    if constexpr (dim == 2)
                        JDet = J(Eigen::all, 0).cross(J(Eigen::all, 1)).stableNorm();
                    else
                        JDet = J.determinant();
                    vInc = 1 * JDet;
                });
            volumeLocal[iCell] = v;
            if (iCell < mesh->NumCell()) // non-ghost
                sumVolume += v;
        }
        real sumVolumeAll{0};
        MPI_Allreduce(&sumVolume, &sumVolumeAll, 1, DNDS_MPI_REAL, MPI_SUM, mpi.comm);
        if (mpi.rank == mRank)
            std::cout
                << "VariationalReconstruction<dim>::ConstructMetrics() === Sum Volume is ["
                << std::setprecision(10) << sumVolumeAll <<"] " << std::endl;
    }

    template void
    VariationalReconstruction<2>::
        ConstructMetrics();
    template void
    VariationalReconstruction<3>::
        ConstructMetrics();
}