#pragma once

#include "DNDS/Defines.hpp"
#include "json.hpp"

namespace DNDS::Geom::RBF
{
    enum RBFKernelType
    {
        UnknownRBFKernel = -1,
        Distance = 0,
        DistanceA1 = 1,
        InversedDistanceA1 = 2,
        Gaussian = 3,
        CPC2 = 4,
    };

    NLOHMANN_JSON_SERIALIZE_ENUM(
        RBFKernelType,
        {{RBFKernelType::UnknownRBFKernel, nullptr},
         {RBFKernelType::Distance, "Distance"},
         {RBFKernelType::DistanceA1, "DistanceA1"},
         {RBFKernelType::InversedDistanceA1, "InversedDistanceA1"},
         {RBFKernelType::Gaussian, "Gaussian"},
         {RBFKernelType::CPC2, "CPC2"}})
}


namespace DNDS::Geom::RBF
{

    inline Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> // redurn Ni at Xj
    RBFCPC2(const tSmallCoords &cent, const tSmallCoords &xs, real R, RBFKernelType kernel = Gaussian)
    {
        Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> RiXj;
        RiXj.resize(cent.cols(), xs.cols());
        for (int iC = 0; iC < cent.cols(); iC++)
        {
            RiXj(iC, Eigen::all) = (xs.colwise() - cent(Eigen::all, iC)).colwise().norm() * (1. / R);
        }
        // Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> NiXj = (1 + RiXj.array().square()).inverse();

        Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> NiXj;
        switch (kernel)
        {
        /*****/ // using (modified) distance Kernel
        case Distance:
            NiXj = RiXj.array();
            break;
        case DistanceA1:
            NiXj = 1 + RiXj.array();
            break;
        case InversedDistanceA1:
            NiXj = (1 + RiXj.array()).inverse();
            break;
        case CPC2:
            /*****/ // using CPC2 Kernel
            NiXj = (1 - RiXj.array()).square().square() * (RiXj.array() * 4 + 1);
            NiXj.array() *= (RiXj.array() < 1.0).cast<real>(); // this kernel is bounded
            break;
        case Gaussian:
            /*****/ // using Gaussian kernel
            NiXj = (-(RiXj.array().square())).exp();
            break;
        default:
            DNDS_assert(false);
        }
        return NiXj;
    }

    template <class TF>
    inline Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>
    RBFInterpolateSolveCoefs(const tSmallCoords &xs, const TF fs, real R, RBFKernelType kernel = Gaussian)
    {
        Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> PT;
        PT.resize(4, xs.cols());
        PT(0, Eigen::all).setConstant(1);
        PT({1, 2, 3}, Eigen::all) = xs;
        Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> M = RBFCPC2(xs, xs, R, kernel);
        Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> A;
        A.setZero(xs.cols() + 4, xs.cols() + 4);
        A.topLeftCorner(xs.cols(), xs.cols()) = M;
        A.bottomLeftCorner(4, xs.cols()) = PT;
        A.topRightCorner(xs.cols(), 4) = PT.transpose();
        Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> RHS;
        RHS.setZero(xs.cols() + 4, fs.cols());
        DNDS_assert(fs.rows() == xs.cols());
        RHS.topRows(xs.cols()) = fs;
        auto LDLT = A.ldlt();
        Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> ret = LDLT.solve(RHS);
        return ret;
    }

    template <class TF>
    inline Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>
    RBFInterpolateSolveCoefsNoPoly(const tSmallCoords &xs, const TF fs, real R, RBFKernelType kernel = Gaussian)
    {
        Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> M = RBFCPC2(xs, xs, R, kernel);
        Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> ret;
        ret.setZero(xs.cols() + 4, fs.cols());
        auto LDLTm = M.ldlt();
        ret.topRows(xs.cols()) = LDLTm.solve(fs);

        return ret;
    }
}