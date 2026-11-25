#pragma once

#include "Physics/RANS/Point.hpp"
#include "Physics/Thermodynamics/ConservativeConversion.hpp"
#include "DNDS/Core/NumericalStatus.hpp"

namespace DNDS::Physics::RANS
{
    DNDS_DEVICE_CALLABLE inline TurbulentTransport TransportFromViscosity(const Point &point, real viscosity,
                                                                          real prandtl, real schmidt) noexcept
    {
        return {viscosity, point.incompressible ? 0 : point.cp * viscosity / prandtl,
                point.incompressible ? 0 : viscosity / std::max(point.state[0] * schmidt, verySmallReal)};
    }
    /// Precondition check for the shared point math, independent of allocation
    /// or address space. Config scales are validated at setup, not per point.
    /// Only active conservative components and their gradients are inspected.
    /// Source-specific cell length is checked by TrySource, not by transport.
    DNDS_DEVICE_CALLABLE inline NumericalStatus CheckPoint(const PointParameters &parameters, const Point &point) noexcept
    {
        if (!parameters.Supported())
            return {NumericalError::UnsupportedRequest};
        if (parameters.kind == ModelKind::None)
            return {};
        if (!(point.state[0] > 0 && point.molecularViscosity > 0 && point.wallDistance > 0) ||
            !std::isfinite(point.molecularViscosity) || !std::isfinite(point.wallDistance) ||
            (!point.incompressible && (!(point.cp > 0 && point.gamma > 1) || !std::isfinite(point.cp) || !std::isfinite(point.gamma))))
            return {NumericalError::InvalidState, 1};
        for (int j = 0; j < 5 + parameters.ComponentNumber(); ++j)
        {
            if (point.incompressible && j == 4)
                continue; // The mechanical closure has no energy channel.
            if (!std::isfinite(point.state[j]))
                return {NumericalError::InvalidState, 10 + j};
            for (int d = 0; d < 3; ++d)
                if (!std::isfinite(point.gradient(d, j)))
                    return {NumericalError::InvalidState, 20 + 3 * j + d};
        }
        return {};
    }

    /// RANS fraction used by the active hybrid source, sampled at the cell
    /// mean. DDES and SST keep full shielding when the DES length does not
    /// activate. SST's existing hybrid source uses F2, independently of the
    /// SA-only desMode selector.
    DNDS_DEVICE_CALLABLE inline NumericalResult<real> TryDESShield(const PointParameters &parameters, const Point &point) noexcept
    {
        const auto status = CheckPoint(parameters, point);
        if (!status)
            return status;
        if (!(parameters.desScale > 0 && parameters.desScale <= 100 &&
              std::isfinite(point.cellLength) && point.cellLength > 0))
            return NumericalStatus{NumericalError::InvalidInput};
        const real d = std::min(point.wallDistance, std::pow(veryLargeReal, 1. / 6.));
        const real rho = point.state[0], nu = point.molecularViscosity / rho;
        const real lesLength = parameters.desScale * point.cellLength;
        if (parameters.kind == ModelKind::KOmegaSST)
        {
            const real floor = std::pow(verySmallReal, 1. / 4.);
            const real k = std::max(point.state[5] / rho, floor);
            const real omega = std::max(point.state[6] / rho, floor);
            const real f2 = SSTF2(k, omega, nu, point.wallDistance);
            return std::sqrt(k) / (.09 * omega) * (1 - f2) > lesLength ? f2 : 1.0;
        }
        if (parameters.kind != ModelKind::SA)
            return NumericalStatus{NumericalError::UnsupportedRequest};
        if (parameters.desMode == SAHybridMode::DES97)
            return d <= lesLength ? 1.0 : 0.0;

        const real chi = point.state[5] * parameters.viscosityReference / point.molecularViscosity;
        const real chi3 = cube(chi);
        const real fnu1 = chi3 / (chi3 + cube(SA::cnu1));
        const real nuTur = nu * std::max(chi * fnu1, 0.0);
        const Eigen::Vector3d velocity = point.state.template segment<3>(1) / rho;
        const Eigen::Matrix3d velocityGradient =
            (point.gradient.template block<3, 3>(0, 1) - point.gradient.col(0) * velocity.transpose()) / rho;
        const real denominator = SADESDenominator(d, velocityGradient.norm());
        if (parameters.desMode == SAHybridMode::DDES)
        {
            const real ft2 = parameters.sa.useFt2 ? 1.2 * std::exp(-.5 * sqr(chi)) : 0.0;
            const real psi = SADESPsi(chi, fnu1, 1 - chi / (1 + chi * fnu1), ft2);
            return d <= lesLength * psi ? 1.0 : SADDESShield((nuTur + nu) / denominator);
        }
        return SAIDDESShield(.25 - d / point.cellLength, nuTur / denominator, parameters.desMode);
    }

    /// Euler turbulent transport formulas. The false diagnostic template
    /// arguments suppress legacy fatal printing; outputs are checked here and
    /// failures returned explicitly. No clipping or alternate closure is added.
    /// A nonnegative prepared viscosity supplies a frozen closure coefficient.
    DNDS_DEVICE_CALLABLE inline NumericalResult<TurbulentTransport> TryTransport(const PointParameters &parameters, const Point &point,
                                                                                 real preparedViscosity = -1) noexcept
    {
        const auto status = CheckPoint(parameters, point);
        if (!status)
            return status;
        real viscosity = preparedViscosity;
        using State = const Eigen::Matrix<real, 7, 1> &;
        using Gradient = const Eigen::Matrix<real, 3, 7> &;
        if (viscosity < 0)
        {
            switch (parameters.kind)
            {
            case ModelKind::SA:
                viscosity = GetMut_SA<3>(point.state, parameters.viscosityReference, point.molecularViscosity);
                break;
            case ModelKind::KOmegaWilcox:
                viscosity = GetMut_KOWilcox<3, State, Gradient, false>(point.state, point.gradient, point.molecularViscosity, point.wallDistance, parameters.kOmega);
                break;
            case ModelKind::KOmegaSST:
                viscosity = GetMut_SST<3, State, Gradient, false>(point.state, point.gradient, point.molecularViscosity, point.wallDistance, parameters.sst);
                break;
            default:
                return TurbulentTransport{};
            }
        }
        const auto result = TransportFromViscosity(point, viscosity, parameters.turbulentPrandtl, parameters.turbulentSchmidt);
        if (!std::isfinite(result.viscosity) || !std::isfinite(result.conductivity) || !std::isfinite(result.speciesDiffusivity))
            return NumericalStatus{NumericalError::InvalidState, 50};
        return result;
    }

    /// Conservative-to-specific chain rule followed by Euler normal diffusion.
    /// Caller supplies a unit normal and molecular+turbulent point properties.
    /// Energy is not interpreted as a temperature/pressure gradient here.
    DNDS_DEVICE_CALLABLE inline NumericalResult<std::array<real, 2>> TryDiffusionFlux(
        const PointParameters &parameters, const Point &point, const Eigen::Vector3d &normal, real turbulentViscosity) noexcept
    {
        const auto status = CheckPoint(parameters, point);
        if (!status)
            return status;
        if (parameters.kind == ModelKind::None)
            return std::array<real, 2>{};
        if (!std::isfinite(turbulentViscosity) || turbulentViscosity < 0)
            return NumericalStatus{NumericalError::InvalidState};
        for (int d = 0; d < 3; ++d)
            if (!std::isfinite(normal[d]))
                return NumericalStatus{NumericalError::InvalidInput};
        Eigen::Matrix<real, 3, 7> gradient = Eigen::Matrix<real, 3, 7>::Zero();
        gradient.col(0) = point.gradient.col(0);
        for (int j = 1; j < 7; ++j)
            if (j != 4)
                for (int d = 0; d < 3; ++d)
                    gradient(d, j) = Thermodynamics::ConservativeSpecificDerivative(point.state[0], point.state[j] / point.state[0],
                                                                                    point.gradient(d, 0), point.gradient(d, j));
        Eigen::Matrix<real, 7, 1> flux = Eigen::Matrix<real, 7, 1>::Zero();
        using State = const Eigen::Matrix<real, 7, 1> &;
        using Gradient = Eigen::Matrix<real, 3, 7> &;
        using Normal = const Eigen::Vector3d &;
        using Flux = Eigen::Matrix<real, 7, 1>;
        switch (parameters.kind)
        {
        case ModelKind::SA:
            GetVisFlux_SA<3>(point.state, gradient, normal, parameters.viscosityReference, point.molecularViscosity, flux);
            break;
        case ModelKind::KOmegaWilcox:
            GetVisFlux_KOWilcox<3>(point.state, gradient, normal, turbulentViscosity, point.wallDistance, point.molecularViscosity, flux, parameters.kOmega);
            break;
        case ModelKind::KOmegaSST:
            GetVisFlux_SST<3, State, Normal, Gradient, Flux, false>(point.state, gradient, normal, turbulentViscosity, point.wallDistance, point.molecularViscosity, flux, parameters.sst);
            break;
        default:
            break;
        }
        if (!std::isfinite(flux[5]) || !std::isfinite(flux[6]))
            return NumericalStatus{NumericalError::InvalidState};
        return std::array<real, 2>{flux[5], flux[6]};
    }

    /// Euler source or positive destruction diagonal, with all conservative
    /// conversion/scaling retained. Mode 0 returns S; mode 1 returns -dS/dU's
    /// established destruction approximation (not a full Jacobian). SST and
    /// enabled SA DES require explicit positive cellLength. No exception, log,
    /// allocation, repair or host fallback occurs in this function.
    DNDS_DEVICE_CALLABLE inline NumericalResult<std::array<real, 2>> TrySource(
        const PointParameters &parameters, const Point &point, int mode) noexcept
    {
        const auto status = CheckPoint(parameters, point);
        if (!status)
            return status;
        if (mode != 0 && mode != 1)
            return NumericalStatus{NumericalError::InvalidInput};
        if (((parameters.kind == ModelKind::SA && parameters.desScale <= 100) || parameters.kind == ModelKind::KOmegaSST) &&
            !(std::isfinite(point.cellLength) && point.cellLength > 0))
            return NumericalStatus{NumericalError::InvalidInput};
        Eigen::Matrix<real, 7, 1> source = Eigen::Matrix<real, 7, 1>::Zero();
        using State = const Eigen::Matrix<real, 7, 1> &;
        using Gradient = const Eigen::Matrix<real, 3, 7> &;
        using Source = Eigen::Matrix<real, 7, 1>;
        switch (parameters.kind)
        {
        case ModelKind::SA:
            GetSource_SA<3, State, Gradient, Source, false>(point.state, point.gradient, parameters.viscosityReference, point.molecularViscosity, point.gamma,
                                                            std::min(point.wallDistance, std::pow(veryLargeReal, 1. / 6.)), parameters.desScale, point.cellLength, parameters.desMode,
                                                            source, parameters.saRotationCorrection, mode, parameters.sa, parameters.saVersion);
            break;
        case ModelKind::KOmegaWilcox:
            GetSource_KOWilcox<3>(point.state, point.gradient, point.molecularViscosity, point.wallDistance, source, mode, parameters.kOmega);
            break;
        case ModelKind::KOmegaSST:
            GetSource_SST<3>(point.state, point.gradient, point.molecularViscosity, point.wallDistance,
                             point.cellLength * parameters.desScale, source, mode, parameters.sst, parameters.sstProduction);
            break;
        default:
            break;
        }
        if (!std::isfinite(source[5]) || !std::isfinite(source[6]))
            return NumericalStatus{NumericalError::InvalidState};
        return std::array<real, 2>{source[5], source[6]};
    }
}
