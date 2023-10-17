#pragma once

#include "Euler.hpp"

namespace DNDS::Euler::RANS
{
    template <int dim, class TU, class TDiffU>
    real GetMut_RealizableKe(TU &&UMeanXy, TDiffU &&DiffUxy, real muf)
    {
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto I4 = dim + 1;
        static const auto verySmallReal_3 = std::pow(verySmallReal, 1. / 3);
        static const auto verySmallReal_4 = std::pow(verySmallReal, 1. / 4);
        real cmu = 0.09;
        real phi = 2. / 3.;
        Eigen::Matrix<real, dim, 1> velo = UMeanXy(Seq123) / UMeanXy(0);
        Eigen::Matrix<real, dim, 1> diffRho = DiffUxy(Seq012, {0});
        Eigen::Matrix<real, dim, dim> diffRhoU = DiffUxy(Seq012, Seq123);
        Eigen::Matrix<real, dim, dim> diffU = (diffRhoU - diffRho * velo.transpose()) / UMeanXy(0);
        Eigen::Matrix<real, dim, dim> SS = diffU + diffU.transpose() - (2. / 3.) * diffU.trace() * Eigen::Matrix<real, dim, dim>::Identity();
        real rho = UMeanXy(0);
        real k = UMeanXy(I4 + 1) / rho + verySmallReal_4;
        real epsilon = UMeanXy(I4 + 2) / rho + verySmallReal_4;
        real Ret = rho * sqr(k) / (muf * epsilon) + smallReal;
        real S = std::sqrt(SS.squaredNorm() / 2) + verySmallReal_4;
        real fmu = (1 - std::exp(-0.01 * Ret)) / (1 - exp(-std::sqrt(Ret))) * std::max(1., std::sqrt(2 / Ret));
        real mut = cmu * fmu * rho * sqr(k) / epsilon;
        mut = std::min(mut, phi * rho * k / S);
        if (std::isnan(mut) || !std::isfinite(mut))
        {
            std::cerr << k << " " << epsilon << " " << Ret << " " << S << "\n";
            std::cerr << fmu << "\n";
            std::cerr << SS << std::endl;
            DNDS_assert(false);
        }
        return mut;
    }

    template <int dim, class TU, class TDiffU, class TVFlux>
    void GetVisFlux_RealizableKe(TU &&UMeanXy, TDiffU &&DiffUxy, real mut, real muPhy, TVFlux &vFlux)
    {
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto I4 = dim + 1;
        real sigK = 1.;
        real sigE = 1.3;
        Eigen::Matrix<real, dim, 2> diffRhoKe = DiffUxy(Seq012, {I4 + 1, I4 + 2});
        Eigen::Matrix<real, dim, 1> diffRho = DiffUxy(Seq012, {0});
        Eigen::Matrix<real, dim, 2> diffKe = (diffRhoKe - 1. / UMeanXy(0) * diffRho * UMeanXy({I4 + 1, I4 + 2}).transpose()) / UMeanXy(0);
        vFlux(Seq012, {I4 + 1}) = diffKe(Seq012, 0) * (muPhy + mut / sigK);
        vFlux(Seq012, {I4 + 2}) = diffKe(Seq012, 1) * (muPhy + mut / sigE);
    }

    template <int dim, class TU, class TDiffU, class TSource>
    void GetSource_RealizableKe(TU &&UMeanXy, TDiffU &&DiffUxy, real muf, TSource &source)
    {
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto I4 = dim + 1;

        real cmu = 0.09;
        real phi = 2. / 3.;
        real ce1 = 1.44;
        real ce2 = 1.92;
        real AE = 0.3;
        real pphi = 1.065;
        Eigen::Matrix<real, dim, 1> velo = UMeanXy(Seq123) / UMeanXy(0);
        Eigen::Matrix<real, dim, 1> diffRho = DiffUxy(Seq012, {0});
        Eigen::Matrix<real, dim, dim> diffRhoU = DiffUxy(Seq012, Seq123);
        Eigen::Matrix<real, dim, dim> diffU = (diffRhoU - diffRho * velo.transpose()) / UMeanXy(0);
        Eigen::Matrix<real, dim, dim> SS = diffU + diffU.transpose() - (2. / 3.) * diffU.trace() * Eigen::Matrix<real, dim, dim>::Identity();
        real rho = UMeanXy(0);
        real k = UMeanXy(I4 + 1) / rho + verySmallReal;
        real epsilon = UMeanXy(I4 + 2) / rho + verySmallReal;
        real Ret = rho * sqr(k) / (muf * epsilon) + verySmallReal;
        real S = std::sqrt(SS.squaredNorm() / 2) + verySmallReal;
        real fmu = (1 - std::exp(-0.01 * Ret)) / (1 - exp(-std::sqrt(Ret))) * std::max(1., std::sqrt(2 / Ret));
        real mut = cmu * fmu * rho * sqr(k) / epsilon;
        mut = std::min(mut, phi * rho * k / S);

        Eigen::Matrix<real, dim, dim> rhoMuiuj = Eigen::Matrix<real, dim, dim>::Identity() * UMeanXy(I4 + 1) * (2. / 3.) - mut * SS;
        real Pk = -(rhoMuiuj.array() * diffU.array()).sum();
        Pk = std::min(Pk, pphi * cmu * sqr(UMeanXy(I4 + 1)) / mut);
        real zeta = std::sqrt(Ret / 2);
        real Tt = k / epsilon * std::max(1., 1. / zeta) + smallReal;

        Eigen::Matrix<real, dim, 2> diffRhoKe = DiffUxy(Seq012, {I4 + 1, I4 + 2});
        Eigen::Matrix<real, dim, 2> diffKe = (diffRhoKe - 1. / UMeanXy(0) * diffRho * UMeanXy({I4 + 1, I4 + 2}).transpose()) / UMeanXy(0);
        Eigen::Matrix<real, dim, 1> diffTau = diffKe(Seq012, 0) / epsilon - k / sqr(epsilon) * diffKe(Seq012, 1);
        real Psi = std::max(0., diffKe(Seq012, 0).dot(diffTau));
        real E = AE * rho * std::sqrt(epsilon * Tt) * Psi * std::max(std::sqrt(k), std::pow(muf * epsilon / rho, 0.25));

        source(I4 + 1) = Pk - UMeanXy(I4 + 2);
        source(I4 + 2) = (ce1 * Pk - ce2 * UMeanXy(I4 + 2) + E) / Tt;
    }

    template <int dim, class TU, class TDiffU, class TSource>
    void GetSourceJacobianDiag_RealizableKe(TU &&UMeanXy, TDiffU &&DiffUxy, real muf, TSource &source)
    {
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto I4 = dim + 1;

        real cmu = 0.09;
        real phi = 2. / 3.;
        real ce1 = 1.44;
        real ce2 = 1.92;
        real AE = 0.3;
        real pphi = 1.065;
        Eigen::Matrix<real, dim, 1> velo = UMeanXy(Seq123) / UMeanXy(0);
        Eigen::Matrix<real, dim, 1> diffRho = DiffUxy(Seq012, {0});
        Eigen::Matrix<real, dim, dim> diffRhoU = DiffUxy(Seq012, Seq123);
        Eigen::Matrix<real, dim, dim> diffU = (diffRhoU - diffRho * velo.transpose()) / UMeanXy(0);
        Eigen::Matrix<real, dim, dim> SS = diffU + diffU.transpose() - (2. / 3.) * diffU.trace() * Eigen::Matrix<real, dim, dim>::Identity();
        real rho = UMeanXy(0);
        real k = UMeanXy(I4 + 1) / rho + verySmallReal;
        real epsilon = UMeanXy(I4 + 2) / rho + verySmallReal;
        real Ret = rho * sqr(k) / (muf * epsilon) + verySmallReal;
        real S = std::sqrt(SS.squaredNorm() / 2) + verySmallReal;
        real fmu = (1 - std::exp(-0.01 * Ret)) / (1 - exp(-std::sqrt(Ret))) * std::max(1., std::sqrt(2 / Ret));
        real mut = cmu * fmu * rho * sqr(k) / epsilon;
        mut = std::min(mut, phi * rho * k / S);

        Eigen::Matrix<real, dim, dim> rhoMuiuj = Eigen::Matrix<real, dim, dim>::Identity() * UMeanXy(I4 + 1) * (2. / 3.) - mut * SS;
        real Pk = -(rhoMuiuj.array() * diffU.array()).sum();
        Pk = std::min(Pk, pphi * cmu * sqr(UMeanXy(I4 + 1)) / mut);
        real zeta = std::sqrt(Ret / 2);
        real Tt = k / epsilon * std::max(1., 1. / zeta) + smallReal;

        Eigen::Matrix<real, dim, 2> diffRhoKe = DiffUxy(Seq012, {I4 + 1, I4 + 2});
        Eigen::Matrix<real, dim, 2> diffKe = (diffRhoKe - 1. / UMeanXy(0) * diffRho * UMeanXy({I4 + 1, I4 + 2}).transpose()) / UMeanXy(0);
        Eigen::Matrix<real, dim, 1> diffTau = diffKe(Seq012, 0) / epsilon - k / sqr(epsilon) * diffKe(Seq012, 1);
        real Psi = std::max(0., diffKe(Seq012, 0).dot(diffTau));
        real E = AE * rho * std::sqrt(epsilon * Tt) * Psi * std::max(std::sqrt(k), std::pow(muf * epsilon / rho, 0.25));

        real dPk = -((Eigen::Matrix<real, dim, dim>::Identity() * (2. / 3.)).array() * diffU.array()).sum();

        source(I4 + 1) = -Pk / (UMeanXy(I4 + 1) + verySmallReal);
        source(I4 + 2) = (ce2) / Tt;
    }

    template <int dim, class TU, class TDiffU, class TSource>
    void GetSourceJacobianDiag_RealizableKe_ND(TU &&UMeanXy, TDiffU &&DiffUxy, real muf, TSource &source)
    {
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto I4 = dim + 1;
        Eigen::VectorXd u0 = UMeanXy;
        Eigen::VectorXd u1 = UMeanXy;
        Eigen::VectorXd sb = source;
        Eigen::VectorXd s0 = source;
        Eigen::VectorXd s1 = source;
        real epsRk = (u0(I4 + 1) + smallReal) * std::sqrt(smallReal);
        real epsRe = (u0(I4 + 2) + smallReal) * std::sqrt(smallReal);
        u0(I4 + 1) += epsRk;
        u1(I4 + 2) += epsRe;
        GetSource_RealizableKe<dim>(UMeanXy, DiffUxy, muf, sb);
        GetSource_RealizableKe<dim>(u0, DiffUxy, muf, s0);
        GetSource_RealizableKe<dim>(u1, DiffUxy, muf, s1);
        source(I4 + 1) = -(s0(I4 + 1) - sb(I4 + 1)) / epsRk;
        source(I4 + 2) = -(s1(I4 + 2) - sb(I4 + 2)) / epsRe;
        source(I4 + 1) = std::max(source(I4 + 1), 0.) * 10;
        source(I4 + 2) = std::max(source(I4 + 2), 0.) * 10;
    }
}