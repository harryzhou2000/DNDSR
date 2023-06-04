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
    using namespace DNDS::Geom;
    using namespace DNDS::Geom::Elem;
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
    using namespace DNDS::Geom;
    using namespace DNDS::Geom::Elem;
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
                    Eigen::Matrix3d J = NodePParams * DiNj({1, 2, 3}, Eigen::all).transpose();
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

void test_Faces()
{
    using namespace DNDS::Geom;
    using namespace DNDS::Geom::Elem;
    for (int i = 1; i < ElemType_NUM; i++)
    {
        auto elem = Element{ElemType(i)};
        auto NodePParams = GetStandardCoord(elem.type);
        std::vector<t_index> nodeInds;
        for (int in = 0; in < elem.GetNumNodes(); in++)
            nodeInds.push_back(in);
        for (int iFace = 0; iFace < elem.GetNumFaces(); iFace++)
        {
            tPoint NormVec;
            bool NormGot = false;
            auto f_elem = elem.ObtainFace(iFace);
            std::vector<t_index> f_nodeInds(f_elem.GetNumNodes());
            elem.ExtractFaceNodes(iFace, nodeInds, f_nodeInds);
            Eigen::Matrix3Xd f_NodePParams = NodePParams(Eigen::all, f_nodeInds);
            for (int order = 0; order <= INT_ORDER_MAX; order++)
            {
                auto quad = Quadrature(f_elem, order);
                double vol = 0;
                quad.Integration(
                    vol,
                    [&](double &inc, int iG, const tPoint &pParam, auto &DiNj)
                    {
                        tPoint pP1 = f_NodePParams * DiNj(0, Eigen::all).transpose();

                        Eigen::Matrix3d J = f_NodePParams * DiNj({1, 2, 3}, Eigen::all).transpose();
                        // std::cout << J << std::endl;
                        tPoint normVec;
                        if (elem.GetDim() == 3)
                            normVec = FacialJacobianToNormVec<3>(J);
                        if (elem.GetDim() == 2)
                            normVec = FacialJacobianToNormVec<2>(J);

                        double tol = 1e-14;
                        if (elem.GetParamSpace() == PyramidSpace)
                            tol *= elem.GetNumNodes();

                        if (NormGot)
                            assert((NormVec - normVec).norm() < tol);
                        NormVec = normVec;
                        NormGot = true;
                        auto SEQDIM = Eigen::seq(0, elem.GetDim() - 1);
                        auto D = J(SEQDIM, SEQDIM).determinant();
                        inc = 1 * D;
                    });
            }
            std::cout << "Norm at Elem " << elem.type
                      << " Face " << iFace
                      << ": " << NormVec.transpose().normalized() << std::endl;
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
    test_Faces();

    // MPI_Finalize();

    return 0;
}