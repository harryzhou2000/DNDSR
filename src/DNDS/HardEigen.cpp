#include "HardEigen.hpp"
#include <iostream>

namespace DNDS::HardEigen
{
    /// @todo test these eigen solvers !!
    // #define EIGEN_USE_LAPACKE
    real EigenLeastSquareInverse(const Eigen::MatrixXd &A, Eigen::MatrixXd &AI)
    {
        // static const double sVmin = 1e-12;
        // auto SVDResult = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
        auto SVDResult = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
        AI = SVDResult.solve(Eigen::MatrixXd::Identity(A.rows(), A.rows()));
        Eigen::VectorXd svs = SVDResult.singularValues().array().abs();
        return svs.maxCoeff() / (svs.minCoeff() + verySmallReal);
    }

    real EigenLeastSquareInverse_Filtered(const Eigen::MatrixXd &A, Eigen::MatrixXd &AI)
    {
        static const double sVmin = 1e-12;
        auto SVDResult = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
        // auto SVDResult = A.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);

        // AI = SVDResult.solve(Eigen::MatrixXd::Identity(A.rows(), A.rows()));
        auto sVs = SVDResult.singularValues();
        real sVsMax = SVDResult.singularValues().array().abs().maxCoeff();
        for (auto &i : sVs)
            if (std::fabs(i) > sVmin * sVsMax) //! note this filtering!
                i = 1. / i;
            else
                i = 0.;
        AI = SVDResult.matrixV() * sVs.asDiagonal() * SVDResult.matrixU().transpose();
        Eigen::VectorXd svs = SVDResult.singularValues().array().abs();
        return svs.maxCoeff() / (svs.minCoeff() + verySmallReal);
        // std::cout << AI * A << std::endl;
    }

    Eigen::Matrix3d Eigen3x3RealSymEigenDecomposition(const Eigen::Matrix3d &A)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver;
        solver.computeDirect(A);
        // if (!(solver.eigenvalues()(1) <= solver.eigenvalues()(2)))
        // {
        //     std::cout << solver.eigenvalues() << std::endl;
        //     std::exit(-1);
        // }
        Eigen::Matrix3d ret = (solver.eigenvectors() * solver.eigenvalues().array().abs().sqrt().matrix().asDiagonal())(Eigen::all, {2, 1, 0});
        // ret(Eigen::all, 0) *= signP(ret(0, 0));
        // ret(Eigen::all, 1) *= signP(ret(1, 1));
        // ret(Eigen::all, 2) *= ret.determinant();
        return ret;
    }

    Eigen::Matrix2d Eigen2x2RealSymEigenDecomposition(const Eigen::Matrix2d &A)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver;
        solver.computeDirect(A);
        // if (!(solver.eigenvalues()(1) <= solver.eigenvalues()(2)))
        // {
        //     std::cout << solver.eigenvalues() << std::endl;
        //     std::exit(-1);
        // }
        Eigen::Matrix2d ret = (solver.eigenvectors() * solver.eigenvalues().array().abs().sqrt().matrix().asDiagonal())(Eigen::all, {1, 0});
        // ret(Eigen::all, 0) *= signP(ret(0, 0));
        // ret(Eigen::all, 1) *= signP(ret(1, 1));
        // ret(Eigen::all, 2) *= ret.determinant();
        return ret;
    }

    real Eigen2x2RealSymEigenDecompositionGetCond(const Eigen::Matrix2d &A)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver;
        solver.computeDirect(A);
        Eigen::Vector2d ev = solver.eigenvalues().array().abs();
        return ev.maxCoeff() / (ev.minCoeff() + verySmallReal);
    }

    real Eigen3x3RealSymEigenDecompositionGetCond(const Eigen::Matrix3d &A)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver;
        solver.computeDirect(A);
        Eigen::Vector3d ev = solver.eigenvalues().array().abs();
        return ev.maxCoeff() / (ev.minCoeff() + verySmallReal);
    }

    real Eigen3x3RealSymEigenDecompositionGetCond01(const Eigen::Matrix3d &A)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver;
        solver.computeDirect(A);
        Eigen::Vector3d ev = solver.eigenvalues().array().abs();
        std::sort(ev.begin(), ev.end());
        return ev(2) / (ev(1) + verySmallReal);
    }

    Eigen::Matrix3d Eigen3x3RealSymEigenDecompositionNormalized(const Eigen::Matrix3d &A)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver;
        solver.computeDirect(A);
        // if (!(solver.eigenvalues()(1) <= solver.eigenvalues()(2)))
        // {
        //     std::cout << solver.eigenvalues() << std::endl;
        //     std::exit(-1);
        // }
        Eigen::Matrix3d ret = (solver.eigenvectors())(Eigen::all, {2, 1, 0});
        // ret(Eigen::all, 0) *= signP(ret(0, 0));
        // ret(Eigen::all, 1) *= signP(ret(1, 1));
        // ret(Eigen::all, 2) *= ret.determinant();
        return ret;
    }

    Eigen::Matrix2d Eigen2x2RealSymEigenDecompositionNormalized(const Eigen::Matrix2d &A)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver;
        solver.computeDirect(A);
        // if (!(solver.eigenvalues()(0) <= solver.eigenvalues()(1)))
        // {
        //     std::cout << solver.eigenvalues() << std::endl;
        //     std::exit(-1);
        // }
        Eigen::Matrix2d ret = (solver.eigenvectors())(Eigen::all, {1, 0});
        // ret(Eigen::all, 0) *= signP(ret(0, 0));
        // ret(Eigen::all, 1) *= ret.determinant();
        return ret;
    }

    Eigen::Index EigenLeastSquareSolve(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, Eigen::MatrixXd &AIB)
    {
        auto SVDResult = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
        AIB = SVDResult.solve(B);
        return SVDResult.rank();
    }

}