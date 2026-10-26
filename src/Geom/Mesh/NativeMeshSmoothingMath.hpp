#pragma once

#include "Geom/Geometric.hpp"
#include "Geom/Quadrature.hpp"

namespace DNDS::Geom::NativeSmoothingDetail
{
    // Established Hermite and bisect-FEM formulas shared by native setup and
    // the explicit legacy smoothing adapters. Numerical conventions retained.
    inline tPoint LineMidpoint(tPoint c0, tPoint c1, tPoint n0, tPoint n1)
    {
        tPoint delta = c1 - c0;
        tPoint unit = delta.stableNormalized();
        real length = delta.stableNorm();
        tPoint t0 = -unit.cross(n0).cross(n0), t1 = -unit.cross(n1).cross(n1);
        t0 = t0.stableNormalized() * length;
        t1 = t1.stableNormalized() * length;
        if (n0.norm() == 0 && n1.norm() != 0)
            return .25 * c0 + .75 * c1 - .25 * t1;
        if (n0.norm() != 0 && n1.norm() == 0)
            return .75 * c0 + .25 * c1 + .25 * t0;
        if (n0.norm() == 0 && n1.norm() == 0)
            return .5 * (c0 + c1);
        return .5 * (c0 + c1) + .125 * (t0 - t1);
    }
    inline tPoint QuadMidpoint(tPoint c0, tPoint c1, tPoint c2, tPoint c3,
                               tPoint n0, tPoint n1, tPoint n2, tPoint n3)
    {
        return .5 * (LineMidpoint(c0, c1, n0, n1) + LineMidpoint(c1, c2, n1, n2) + LineMidpoint(c2, c3, n2, n3) + LineMidpoint(c3, c0, n3, n0)) - .25 * (c0 + c1 + c2 + c3);
    }
    inline MatrixXR Stiffness(Elem::Element eCell, const SmallCoordsAsVector &coordsC, const SmallCoordsAsVector &coordsCu, int dim)
    {
        const rowsize nnLoc = rowsize(coordsC.cols());
        MatrixXR ALoc;
        ALoc.setZero(static_cast<index>(nnLoc) * 3, static_cast<index>(nnLoc) * 3 + 1);
        auto localNodeIdx = Eigen::ArrayXi::LinSpaced(nnLoc, 0, nnLoc - 1);

        // Bisect the O2 element into O1 sub-elements.
        int nBi = eCell.GetO2NumBisect();
        int iBiVariant = GetO2ElemBisectVariant(eCell, coordsC);
        for (int iBi = 0; iBi < nBi; iBi++)
        {
            auto eCellSub = eCell.ObtainO2BisectElem(iBi);
            Eigen::ArrayXi c2nSubLocal;
            c2nSubLocal.resize(eCellSub.GetNumNodes());
            eCell.ExtractO2BisectElemNodes(iBi, iBiVariant, localNodeIdx, c2nSubLocal);
            SmallCoordsAsVector coordsCSub, coordsCuSub;
            coordsCSub.resize(3, eCellSub.GetNumNodes());
            coordsCuSub.resize(3, eCellSub.GetNumNodes());
            eCell.ExtractO2BisectElemNodes(iBi, iBiVariant, coordsC, coordsCSub);
            eCell.ExtractO2BisectElemNodes(iBi, iBiVariant, coordsCu, coordsCuSub);
            auto qCellSub = Elem::Quadrature{eCellSub, 6};
            auto nnLocSub = rowsize(c2nSubLocal.size());
            MatrixXR ALocSub, mLoc;
            ALocSub.setZero(static_cast<index>(nnLocSub) * 3, static_cast<index>(nnLocSub) * 3 + 1);
            mLoc.resize(6, static_cast<index>(nnLocSub) * 3);
            Eigen::ArrayXi c2nSubLocal3;
            c2nSubLocal3.resize(c2nSubLocal.size() * 3);
            c2nSubLocal3(Eigen::seq(c2nSubLocal.size() * 0, c2nSubLocal.size() * 1 - 1)) = c2nSubLocal + nnLoc * 0;
            c2nSubLocal3(Eigen::seq(c2nSubLocal.size() * 1, c2nSubLocal.size() * 2 - 1)) = c2nSubLocal + nnLoc * 1;
            c2nSubLocal3(Eigen::seq(c2nSubLocal.size() * 2, c2nSubLocal.size() * 3 - 1)) = c2nSubLocal + nnLoc * 2;

            // Integrate elasticity stiffness on the sub-element.
            qCellSub.Integration(
                ALocSub,
                [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                {
                    auto getJ = [&](auto coos)
                    {
                        tJacobi J = Elem::ShapeJacobianCoordD01Nj(coos, DiNj);
                        tJacobi JInv = tJacobi::Identity();
                        real JDet = UnInitReal;
                        if (dim == 2)
                            JDet = J(EigenAll, 0).cross(J(EigenAll, 1)).stableNorm();
                        else
                            JDet = J.determinant();
                        if (dim == 2)
                            JInv({0, 1}, {0, 1}) = J({0, 1}, {0, 1}).inverse().eval();
                        else
                            JInv = J.inverse().eval();
                        return std::make_tuple(J, JInv, JDet);
                    };
                    auto [J, JInv, JDet] = getJ(coordsCSub);
                    auto [Jn, JInvn, JDetn] = getJ(coordsCSub + coordsCuSub);
                    DNDS_check_throw_info(std::isfinite(JDet) && JDet > 0, "native elasticity subelement has nonpositive Jacobian");

                    // Strain-displacement matrix (Voigt notation).
                    tSmallCoords m3 = JInv.transpose() * DiNj({1, 2, 3}, EigenAll);
                    mLoc.setZero();
                    mLoc(0, Eigen::seq(nnLocSub * 0, nnLocSub * 0 + nnLocSub - 1)) = m3(0, EigenAll);
                    mLoc(1, Eigen::seq(nnLocSub * 1, nnLocSub * 1 + nnLocSub - 1)) = m3(1, EigenAll);
                    mLoc(2, Eigen::seq(nnLocSub * 2, nnLocSub * 2 + nnLocSub - 1)) = m3(2, EigenAll);
                    mLoc(3, Eigen::seq(nnLocSub * 1, nnLocSub * 1 + nnLocSub - 1)) = 0.5 * m3(2, EigenAll);
                    mLoc(3, Eigen::seq(nnLocSub * 2, nnLocSub * 2 + nnLocSub - 1)) = 0.5 * m3(1, EigenAll);
                    mLoc(4, Eigen::seq(nnLocSub * 2, nnLocSub * 2 + nnLocSub - 1)) = 0.5 * m3(0, EigenAll);
                    mLoc(4, Eigen::seq(nnLocSub * 0, nnLocSub * 0 + nnLocSub - 1)) = 0.5 * m3(2, EigenAll);
                    mLoc(5, Eigen::seq(nnLocSub * 0, nnLocSub * 0 + nnLocSub - 1)) = 0.5 * m3(1, EigenAll);
                    mLoc(5, Eigen::seq(nnLocSub * 1, nnLocSub * 1 + nnLocSub - 1)) = 0.5 * m3(0, EigenAll);

                    // Constitutive matrix (isotropic linear elasticity).
                    Eigen::Matrix<real, 6, 6> DStruct;
                    DStruct.setIdentity();
                    real lam = 1;
                    real muu = 100;
                    DStruct *= muu;
                    DStruct({0, 1, 2}, {0, 1, 2}) += muu * tJacobi::Identity();
                    DStruct({0, 1, 2}, {0, 1, 2}).array() += lam;
                    DStruct *= std::pow(JDetn / JDet, -0.0) + 1;

                    vInc.resizeLike(ALocSub);
                    vInc(EigenAll, Eigen::seq(0, 3 * nnLocSub - 1)) = mLoc.transpose() * DStruct * mLoc;
                    vInc(EigenAll, EigenLast).setZero();
                    vInc *= JDet;
                });
            ALoc(c2nSubLocal3, c2nSubLocal3) += ALocSub(EigenAll, Eigen::seq(0, EigenLast - 1));
            ALoc(c2nSubLocal3, EigenLast) += ALocSub(EigenAll, EigenLast);
        }

        // Zero out z-component rows/cols for 2D problems.
        if (dim != 3)
        {
            ALoc(Eigen::seq(2 * nnLoc, 3 * nnLoc - 1), Eigen::seq(2 * nnLoc, 3 * nnLoc - 1)).setZero();
            ALoc(EigenAll, Eigen::seq(2 * nnLoc, 3 * nnLoc - 1)).setZero();
            ALoc(Eigen::seq(2 * nnLoc, 3 * nnLoc - 1), EigenAll).setZero();
        }

        return ALoc;
    }
}
