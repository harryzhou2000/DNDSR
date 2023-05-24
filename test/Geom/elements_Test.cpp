#include "Geom/Quadrature.hpp"
#include "Eigen/Dense"
#include <cstdlib>
#include <cassert>

std::vector<double> argD;

/*
running:
valgrind --log-file=log_valgrind.log
*/

void test_ShapeFunc_delta()
{
    using namespace Geom::Elem;
    for (int i = 1; i < ElemType_NUM; i++)
    {
        auto elem = Element{ElemType(i)};
        auto NodePParams = GetStandardCoord(elem.type);
        for (int in = 0; in < elem.GetNumNodes(); in++)
        {
            tNj Nj;
            elem.GetNj(NodePParams(Eigen::all, in), Nj);
            // std::cout << Nj << std::endl;
            double tol = 1e-16;
            if (elem.GetParamSpace() == PyramidSpace && in == 4)
                tol = 1e-14; // relaxed for tip
            assert(std::abs(Nj[in] - 1) < tol);
            Nj[in] -= 1;
            assert(Nj.norm() < tol * elem.GetNumNodes());
        }
    }
}

void test_Standard_Elem()
{
    using namespace Geom::Elem;
    for (int i = 1; i < ElemType_NUM; i++)
    {
        auto elem = Element{ElemType(i)};
        auto NodePParams = GetStandardCoord(elem.type);
        for (int order = 0; order <= INT_ORDER_MAX; order++)
        {
            auto quad = Quadrature(elem, order);
            double vol = 0;
            quad.Integration(
                vol,
                [&](double &inc, int iG, const tPoint &pParam, auto &DiNj)
                {
                    tPoint pP1 = NodePParams * DiNj(0, Eigen::all).transpose();
                    assert((pParam - pP1).norm() < 1e-15);
                    Eigen::Matrix3d J = DiNj({1, 2, 3}, Eigen::all) * NodePParams.transpose();
                    Eigen::Vector3d EyeD;
                    EyeD.setZero();
                    auto SEQDIM = Eigen::seq(0, elem.GetDim() - 1);
                    EyeD(SEQDIM).setConstant(1);
                    // std::cout << J << std::endl;
                    // std::cout << (J - Eigen::Matrix3d{EyeD.asDiagonal()}).norm() << std::endl;
                    double tol = 1e-14;
                    if (elem.GetParamSpace() == PyramidSpace)
                        tol *= elem.GetNumNodes();
                    assert((J - Eigen::Matrix3d{EyeD.asDiagonal()}).norm() < tol);
                    auto D = J(SEQDIM, SEQDIM).determinant();
                    inc = 1 * D;
                });

            // std::cout << "vol of Element " << i << ": " << vol << std::endl;
            assert(std::abs(vol - ParamSpaceVol(elem.GetParamSpace())) < 1e-14 * elem.GetNumNodes());
        }
    }
}

int main(int argc, char *argv[])
{
    // ! Disable MPI call to help serial mem check
    // MPI_Init(&argc, &argv);

    for (int i = 1; i < argc; i++)
    {
        double v = std::atof(argv[i]);
        argD.push_back(v);
    }

    test_ShapeFunc_delta();
    test_Standard_Elem();

    // MPI_Finalize();

    return 0;
}