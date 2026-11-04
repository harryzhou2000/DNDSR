#pragma once

#include "Physics/RANS/RANSModels.hpp"
#include <Eigen/Core>
#include <array>

namespace DNDS::Physics::RANS
{
    /// Physical closure controls, independent of mesh, equation allocation and
    /// nonlinear iteration. Existing kernel configurations retain their defaults.
    struct ModelSettings
    {
        std::string model = "none";
        real viscosityReference = 1, turbulentPrandtl = .9, turbulentSchmidt = .9;
        real desScale = veryLargeReal;
        int desMode = 1, saVersion = 0;
        bool saRotationCorrection = true;
        SAConfig sa;
        KOmegaConfig kOmega;
        KOmegaSSTConfig sst;
        RKEConfig realizableKE;
        DNDS_DECLARE_CONFIG(ModelSettings)
        {
            DNDS_FIELD(model, "none, SA, KOmegaWilcox, KOmegaSST, or RealizableKE");
            DNDS_FIELD(viscosityReference, "SA conservative variable scale: nuTilde = rhoNuTilde * viscosityReference / rho, in code units");
            DNDS_FIELD(turbulentPrandtl, "Turbulent heat conductivity uses cp*mu_t/Pr_t");
            DNDS_FIELD(turbulentSchmidt, "Turbulent species diffusivity uses mu_t/(rho*Sc_t)");
            DNDS_FIELD(desScale, "Euler DES scale; greater than 100 disables SA DES, also multiplies SST cell length");
            DNDS_FIELD(desMode, "SA DES mode: 0 DDES, 1 IDDES, 2 WMLES, 3 DES97");
            DNDS_FIELD(saVersion, "Euler SA variant: 0 current, 1 oldest legacy, 9001 historical sign-error reproduction");
            DNDS_FIELD(saRotationCorrection, "Euler SA-R production correction; supplied velocity is absolute");
            config.field_section(&T::sa, "sa", "Spalart-Allmaras kernel controls");
            config.field_section(&T::kOmega, "kOmega", "Wilcox kernel controls");
            config.field_section(&T::sst, "sst", "Menter SST kernel controls");
            config.field_section(&T::realizableKE, "realizableKE", "Realizable k-epsilon kernel controls");
        }
    };

    enum class ModelKind
    {
        None,
        SA,
        KOmegaWilcox,
        KOmegaSST,
        RealizableKE
    };

    /// Bounded CPU point scratch in one Cartesian frame, using three momentum
    /// components even on 2D geometry. Gather logical blocks into [flow5,RANS2];
    /// the unused seventh component for SA is zero. Gradient is
    /// (direction,conservative component). All quantities use common code units.
    /// No pointer to a mesh, Array, EOS worker or persistent cache is retained.
    struct Point
    {
        Eigen::Matrix<real, 7, 1> state = Eigen::Matrix<real, 7, 1>::Zero();
        Eigen::Matrix<real, 3, 7> gradient = Eigen::Matrix<real, 3, 7>::Zero();
        real molecularViscosity = 0, wallDistance = 1, cellLength = 1;
        real cp = 1, gamma = 1.4;
    };
    struct TurbulentTransport
    {
        real viscosity = 0, conductivity = 0, speciesDiffusivity = 0;
    };

    /// Setup-selected CPU closure. Calls the same source/diffusion functions as
    /// Euler, with no EOS, quadrature, wall search, MPI, clipping or allocation.
    /// Caller validates finite/admissible point inputs and positive molecular
    /// viscosity, density, wall distance and cell length before kernel execution.
    /// Numerical reconstruction, final state repair and wall BCs remain outside.
    /// This adapter is CPU-only; it is not an implicit device fallback.
    class Model
    {
        ModelSettings _settings;
        ModelKind _kind;
        std::array<real, 2> EvaluateSource(const Point &point, int mode) const;

    public:
        explicit Model(ModelSettings settings = {});
        ModelKind Kind() const { return _kind; }
        const ModelSettings &Settings() const { return _settings; }
        int ComponentNumber() const { return _kind == ModelKind::None ? 0 : _kind == ModelKind::SA ? 1
                                                                                                   : 2; }
        /// Stable logical component names; this metadata does not prescribe an
        /// Array grouping, storage layout or position relative to species.
        std::array<const char *, 2> ComponentNames() const;
        TurbulentTransport Transport(const Point &point) const;
        /// Positive molecular+turbulent normal diffusion of the RANS variables.
        /// Computes specific-variable gradients by the conservative chain rule.
        /// Normal is unit length. Flow stress/heat/species transport is separate.
        std::array<real, 2> DiffusionFlux(const Point &point, const Eigen::Vector3d &normal,
                                          real turbulentViscosity) const;
        std::array<real, 2> Source(const Point &point) const { return EvaluateSource(point, 0); }
        /// Euler's positive -dS/dU destruction approximation. It is not a full
        /// numerical derivative; do not negate it again when adding to a matrix.
        std::array<real, 2> ImplicitSourceDiagonal(const Point &point) const { return EvaluateSource(point, 1); }
    };
}
