#pragma once

#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "Cauchy/Core/DomainPlacement.hpp"
#include "Cauchy/Riemann/RiemannFlux.hpp"
#include <map>

namespace DNDS::Cauchy
{
    using Json = nlohmann::ordered_json;
    struct EquationSettings
    {
        std::string system = "compressible", eos = "perfectGas", transport = "inviscid", source = "none";
        real gamma = 1.4, gasConstant = 1., viscosity = 0., prandtl = 0.72;
        DNDS_DECLARE_CONFIG(EquationSettings)
        {
            DNDS_FIELD(system, "Coupled equation family");
            DNDS_FIELD(eos, "Thermodynamic closure");
            DNDS_FIELD(transport, "inviscid or NewtonianFourier");
            DNDS_FIELD(source, "Volume source family; none initially");
            DNDS_FIELD(gamma, "Perfect-gas heat capacity ratio");
            DNDS_FIELD(gasConstant, "Specific gas constant in the chosen units");
            DNDS_FIELD(viscosity, "Constant dynamic viscosity");
            DNDS_FIELD(prandtl, "Molecular Prandtl number");
        }
    };
    struct ExecutionSettings
    {
        std::string device = "cpu", layout = "aos", communication = "mpi-direct";
        int hostThreads = 0;
        bool checkAccess = true;
        DNDS_DECLARE_CONFIG(ExecutionSettings)
        {
            DNDS_FIELD(device, "Required numerical backend; no implicit fallback");
            DNDS_FIELD(layout, "Physical flow table: aos, soa, aosoa32");
            DNDS_FIELD(communication, "Payload transport");
            DNDS_FIELD(hostThreads, "CPU worker count; zero means inline execution");
            DNDS_FIELD(checkAccess, "Validate declared scheduled accesses at submission");
        }
    };
    struct MeshSettings
    {
        std::string file;
        int dimension = 2;
        std::map<std::string, int> boundaryIds;
        DNDS_DECLARE_CONFIG(MeshSettings)
        {
            DNDS_FIELD(file, "CGNS input path; an injected mesh needs no file");
            DNDS_FIELD(dimension, "Geometric dimension, independent of flow component count");
            config.field_schema(&T::boundaryIds, "boundaryIds", "Explicit CGNS boundary name to integer zone",
                                []() -> Json
                                { return {{"type", "object"}, {"additionalProperties", {{"type", "integer"}}}}; });
        }
    };
    struct RiemannSettings
    {
        real fixScale = 1, incrementScale = 1, dLambda = 0, referenceMach = .1, pressureStencilMin = 1e300;
        DNDS_DECLARE_CONFIG(RiemannSettings)
        {
            DNDS_FIELD(fixScale, "Entropy and H-correction scale");
            DNDS_FIELD(incrementScale, "Roe contact dissipation scale");
            DNDS_FIELD(dLambda, "Explicit transverse H-correction signal estimate");
            DNDS_FIELD(referenceMach, "Positive AUSM+-up reference Mach cutoff");
            DNDS_FIELD(pressureStencilMin, "AUSMPW+ explicit stencil minimum; large value selects one-dimensional factor");
        }
        Riemann::FluxSettings KernelSettings() const { return {dLambda, fixScale, incrementScale, referenceMach, pressureStencilMin}; }
    };
    struct NumericsSettings
    {
        std::string reconstruction = "gradient", gradient = "leastSquares", flux = "rusanov", limiter = "none";
        int reconstructionIterations = 8;
        real densityFloor = 1e-12, pressureFloor = 1e-12;
        CFV::ReconstructionSettings cfv;
        RiemannSettings riemann;
        DNDS_DECLARE_CONFIG(NumericsSettings)
        {
            DNDS_FIELD(reconstruction, "constant, gradient, or variational reconstruction");
            DNDS_FIELD(gradient, "leastSquares, greenGauss, or ggmp");
            DNDS_FIELD(flux, "Numerical inviscid flux family");
            DNDS_FIELD(limiter, "none or WBAP; positivity is checked separately");
            DNDS_FIELD(reconstructionIterations, "Fixed local VR sweeps per residual");
            DNDS_FIELD(densityFloor, "Strict positive density admissibility floor");
            DNDS_FIELD(pressureFloor, "Strict positive pressure admissibility floor");
            config.field_section(&T::cfv, "cfv", "Independent mathematical CFV configuration");
            config.field_section(&T::riemann, "riemann", "Numerical flux correction controls, independent of EOS");
        }
    };
    struct BoundarySettings
    {
        std::string kind = "extrapolate", thermal = "adiabatic";
        std::array<real, 5> primitive{1, 0, 0, 0, 1};
        std::array<real, 3> velocity{0, 0, 0};
        real temperature = 1;
        DNDS_DECLARE_CONFIG(BoundarySettings)
        {
            DNDS_FIELD(kind, "extrapolate, farfield, slipWall, or noSlipWall");
            DNDS_FIELD(thermal, "Wall thermal law: adiabatic or isothermal");
            DNDS_FIELD(primitive, "Farfield rho,u,v,w,p; three velocities on any geometry");
            DNDS_FIELD(velocity, "Prescribed stationary-boundary wall velocity");
            DNDS_FIELD(temperature, "Positive isothermal wall temperature");
        }
    };
    struct DomainSettings
    {
        EquationSettings equation;
        ExecutionSettings execution;
        MeshSettings mesh;
        NumericsSettings numerics;
        std::array<real, 5> initialPrimitive{1, 0, 0, 0, 1};
        std::map<std::string, BoundarySettings> boundaries;
        DNDS_DECLARE_CONFIG(DomainSettings)
        {
            config.field_section(&T::equation, "equation", "Equation and constitutive laws");
            config.field_section(&T::execution, "execution", "Explicit execution capabilities");
            config.field_section(&T::mesh, "mesh", "Mesh input");
            config.field_section(&T::numerics, "numerics", "Discrete operators");
            DNDS_FIELD(initialPrimitive, "Constant initial rho,u,v,w,p");
            config.field_map_of<BoundarySettings>(&T::boundaries, "boundaries", "Boundary-zone ID to condition");
        }
    };
    struct TimeSettings
    {
        std::string method = "ssprk3";
        real dt = 1e-3;
        index steps = 1;
        DNDS_DECLARE_CONFIG(TimeSettings)
        {
            DNDS_FIELD(method, "Coordinated physical-time method");
            DNDS_FIELD(dt, "Fixed positive physical timestep");
            DNDS_FIELD(steps, "Nonnegative accepted step target");
        }
    };
    struct OutputSettings
    {
        std::string prefix = "cauchy";
        index interval = 1;
        DNDS_DECLARE_CONFIG(OutputSettings)
        {
            DNDS_FIELD(prefix, "Output file prefix");
            DNDS_FIELD(interval, "Accepted-step snapshot interval; zero disables periodic output");
        }
    };
    struct DomainConfig
    {
        DomainRankGroup placement;
        DomainSettings settings;
    };
    struct ConfigOverride
    {
        std::string pointer;
        Json value;
    };
    struct ResolvedExecutionPlan
    {
        std::vector<std::string> kernels;
        /// Enumerate every unavailable branch before domain construction. A
        /// backend's presence alone does not imply its equation/BC kernels exist.
        static ResolvedExecutionPlan Resolve(const DomainConfig &domain);
    };
    /// Immutable resolved root. Profiles contain DomainSettings patches. Resolve
    /// schema defaults -> selected profiles -> root defaults -> domain overrides
    /// -> CLI JSON pointers. Provenance uses pointers into the expanded root.
    /// Runtime clocks/history/controller values never mutate configuration.
    class SimulationConfig
    {
        struct Data
        {
            Json root;
            std::map<std::string, std::string> provenance;
            std::vector<DomainConfig> domains;
            TimeSettings time;
            OutputSettings output;
        };
        std::shared_ptr<const Data> _data;
        explicit SimulationConfig(std::shared_ptr<const Data> data) : _data(std::move(data)) {}

    public:
        static SimulationConfig Resolve(const Json &input, int simulationSize, const std::vector<ConfigOverride> &overrides = {});
        /// Production entry: coordinate local parsing/capability failures, then
        /// prove exact resolved-root/provenance agreement before mesh collectives.
        static SimulationConfig ResolveCollective(const Json &input, MPI_Comm simulation, const std::vector<ConfigOverride> &overrides = {});
        const Json &Root() const { return _data->root; }
        const std::map<std::string, std::string> &Provenance() const { return _data->provenance; }
        const std::vector<DomainConfig> &Domains() const { return _data->domains; }
        const TimeSettings &Time() const { return _data->time; }
        const OutputSettings &Output() const { return _data->output; }
        std::vector<DomainRankGroup> RankGroups() const;
    };
}
