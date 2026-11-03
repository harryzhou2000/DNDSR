#pragma once

#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "CFV/Math/LimitedVR.hpp"
#include "Cauchy/Core/DomainPlacement.hpp"
#include "Cauchy/Riemann/RiemannFlux.hpp"
#include "Cauchy/Motion/RigidMotion.hpp"
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
        bool overlap = true;
        DNDS_DECLARE_CONFIG(ExecutionSettings)
        {
            DNDS_FIELD(device, "Required numerical backend; no implicit fallback");
            DNDS_FIELD(layout, "Physical flow table: aos, soa, aosoa32");
            DNDS_FIELD(communication, "Payload transport");
            DNDS_FIELD(hostThreads, "CPU worker count; zero means inline execution");
            DNDS_FIELD(checkAccess, "Validate declared scheduled accesses at submission");
            DNDS_FIELD(overlap, "Overlap input halo exchanges with owned-only core work; false waits then runs unsplit kernels");
        }
    };
    struct MeshSettings
    {
        std::string file;
        int dimension = 2;
        std::array<real, 3> periodicLengths{0, 0, 0};
        std::map<std::string, int> boundaryIds;
        DNDS_DECLARE_CONFIG(MeshSettings)
        {
            DNDS_FIELD(file, "CGNS input path; an injected mesh needs no file");
            DNDS_FIELD(dimension, "Geometric dimension, independent of flow component count");
            DNDS_FIELD(periodicLengths, "Translation from PERIODIC_n to its donor; zero disables that direction");
            config.field_schema(&T::boundaryIds, "boundaryIds", "Explicit CGNS boundary name to integer zone",
                                []() -> Json
                                { return {{"type", "object"}, {"additionalProperties", {{"type", "integer"}}}}; });
        }
    };
    struct RiemannSettings
    {
        real fixScale = 1, incrementScale = 1, dLambda = 0, referenceMach = .1, pressureStencilMin = 1e300;
        bool useStencil = true;
        DNDS_DECLARE_CONFIG(RiemannSettings)
        {
            DNDS_FIELD(fixScale, "Entropy and H-correction scale");
            DNDS_FIELD(incrementScale, "Roe contact dissipation scale");
            DNDS_FIELD(dLambda, "Transverse H-correction signal floor; exact scalar input when useStencil is false");
            DNDS_FIELD(referenceMach, "Positive AUSM+-up reference Mach cutoff");
            DNDS_FIELD(pressureStencilMin, "AUSMPW+ stencil pressure upper bound; exact scalar input when useStencil is false");
            DNDS_FIELD(useStencil, "Supply cell-face-cell wave/pressure stencils to fluxes that use them; false keeps the explicit scalar inputs");
        }
        Riemann::FluxSettings KernelSettings() const { return {dLambda, fixScale, incrementScale, referenceMach, pressureStencilMin}; }
    };
    struct NumericsSettings
    {
        std::string reconstruction = "gradient", gradient = "leastSquares", flux = "rusanov", limiter = "none";
        int reconstructionIterations = 8;
        real reconstructionTolerance = 0;
        bool characteristicLimiter = true, faceFallback = true;
        bool reconstructionBoundaryFromPolynomial = true;
        std::string viscousGradient = "primitive";
        bool viscousLimited = true, noGRPOnWall = false;
        bool positivity = false;
        real temperatureFloor = 1e-12;
        real positivityBetaCutoff = 1e-3;
        std::string positivityReconstruction = "Euler";
        bool positivityCompressToMean = true, positivityVolumePoints = true;
        int positivityBetaPower = 1;
        real densityFloor = 1e-12, pressureFloor = 1e-12;
        CFV::ReconstructionSettings cfv;
        CFV::LimitedVRSettings limitedVR;
        RiemannSettings riemann;
        void ValidatePositivity() const;
        DNDS_DECLARE_CONFIG(NumericsSettings)
        {
            DNDS_FIELD(reconstruction, "constant, gradient, or variational reconstruction");
            DNDS_FIELD(gradient, "leastSquares, greenGauss, or ggmp");
            DNDS_FIELD(flux, "Numerical inviscid flux family");
            DNDS_FIELD(limiter, "none, WBAP or native LVR; positivity is checked separately");
            DNDS_FIELD(reconstructionIterations, "Fixed local VR sweeps per residual");
            DNDS_FIELD(reconstructionTolerance, "Relative Jacobi update tolerance; zero uses the fixed sweep count");
            DNDS_FIELD(characteristicLimiter, "Use equation-provided cell-mean face characteristics for 3WBAP");
            DNDS_FIELD(faceFallback, "Fall back to the cell mean at inadmissible face values; disable for non-PP accuracy studies");
            DNDS_FIELD(reconstructionBoundaryFromPolynomial, "Project current face polynomial through BC each VR sweep; false uses mean-based boundary values");
            DNDS_FIELD(viscousGradient, "primitive retains the cell-gradient scheme; polynomial uses cached polynomial derivatives and Euler's normal face-jump penalty");
            DNDS_FIELD(viscousLimited, "Use the filtered flux polynomial for polynomial viscosity; false uses the unlimited VR guess");
            DNDS_FIELD(noGRPOnWall, "Disable the polynomial normal jump penalty on no-slip walls, retaining the polynomial gradient");
            DNDS_FIELD(positivity, "Mean-preserving reconstruction and conservative SSPRK flux-blend PP");
            DNDS_FIELD(temperatureFloor, "Minimum temperature enforced through the equation's internal-energy bound");
            DNDS_FIELD(positivityBetaCutoff, "Set numerically tiny reconstruction beta to zero; matches legacy's mean fallback");
            DNDS_FIELD(positivityReconstruction, "Euler common quadrature beta or Segment for reproducing earlier Cauchy archives");
            DNDS_FIELD(positivityCompressToMean, "Euler reference is the mean; false uses legacy progressive order reduction");
            DNDS_FIELD(positivityVolumePoints, "Check volume as well as face quadrature points; future source models may require volume points");
            DNDS_FIELD(positivityBetaPower, "Euler extra beta exponent, so the total exponent is 1+power");
            DNDS_FIELD(densityFloor, "Positive inclusive density admissibility floor");
            DNDS_FIELD(pressureFloor, "Positive inclusive pressure admissibility floor for the initial perfect-gas equation");
            config.field_section(&T::cfv, "cfv", "Independent mathematical CFV configuration");
            config.field_section(&T::limitedVR, "limitedVR", "Native v2/v3 target-penalty controls");
            config.field_section(&T::riemann, "riemann", "Numerical flux correction controls, independent of EOS");
        }
    };
    struct FlowProfileSettings
    {
        std::string kind = "constant";
        real strength = 5, period = 10;
        std::array<real, 3> center{5, 5, 0};
        real perturbation = 0;
        DNDS_DECLARE_CONFIG(FlowProfileSettings)
        {
            DNDS_FIELD(kind, "constant, isentropicVortex, doubleMachReflection, or cylinderWake");
            DNDS_FIELD(strength, "Isentropic vortex strength; five is the standard non-PP vortex");
            DNDS_FIELD(period, "Periodic vortex box length");
            DNDS_FIELD(center, "Initial vortex center");
            DNDS_FIELD(perturbation, "Localized transverse initial wake perturbation amplitude");
        }
    };
    struct BoundarySettings
    {
        std::string kind = "extrapolate", thermal = "adiabatic";
        std::string wallVelocityFrame = "mesh";
        std::array<real, 5> primitive{1, 0, 0, 0, 1};
        std::array<real, 3> velocity{0, 0, 0};
        real temperature = 1;
        FlowProfileSettings profile;
        DNDS_DECLARE_CONFIG(BoundarySettings)
        {
            DNDS_FIELD(kind, "extrapolate, farfield, slipWall, noSlipWall, analytic, doubleMachBottom, or interface");
            DNDS_FIELD(thermal, "Wall thermal law: adiabatic or isothermal");
            DNDS_FIELD(wallVelocityFrame, "mesh: velocity relative to moving wall; ground: prescribed absolute ground velocity");
            DNDS_FIELD(primitive, "Farfield rho,u,v,w,p; three velocities on any geometry");
            DNDS_FIELD(velocity, "Prescribed wall velocity in wallVelocityFrame; farfield/analytic data always use ground axes");
            DNDS_FIELD(temperature, "Positive isothermal wall temperature");
            config.field_section(&T::profile, "profile", "Space/time input for analytic boundary values");
        }
    };
    struct ImplicitLinearSettings
    {
        std::string method = "gmres", preconditioner = "sgs";
        index iterations = 20;
        int subspace = 10;
        real relativeTolerance = .1, absoluteTolerance = 1e-12;
        DNDS_DECLARE_CONFIG(ImplicitLinearSettings)
        {
            DNDS_FIELD(method, "sgs or restarted left-gmres");
            DNDS_FIELD(preconditioner, "Fixed sgs or rank-local ilu for GMRES");
            DNDS_FIELD(iterations, "Maximum linear iterations per nonlinear correction");
            DNDS_FIELD(subspace, "Restarted GMRES subspace size");
            DNDS_FIELD(relativeTolerance, "Relative true-residual tolerance");
            DNDS_FIELD(absoluteTolerance, "Absolute true-residual tolerance");
        }
    };
    struct ImplicitSGSSettings
    {
        int sweeps = 1;
        bool exchangeBetweenSweeps = false;
        DNDS_DECLARE_CONFIG(ImplicitSGSSettings)
        {
            DNDS_FIELD(sweeps, "Fixed number of zero-started SGS sweeps per GMRES preconditioner application");
            DNDS_FIELD(exchangeBetweenSweeps, "Expose a forward-halo exchange between the lower and upper SGS scans");
        }
    };
    struct ImplicitILUSettings
    {
        int fill = 0;
        bool coreFirst = true;
        DNDS_DECLARE_CONFIG(ImplicitILUSettings)
        {
            DNDS_FIELD(fill, "Local ILU stencil expansion 0..5, or -1 for complete LU");
            DNDS_FIELD(coreFirst, "Order core before dependent cells before symbolic fill");
        }
    };
    struct ImplicitPositivitySettings
    {
        std::string rhsMode = "rhs";
        std::string rhsReduction = "cell";
        real rhsScale = 1, rhsRelax = .5, incrementRelax = 1;
        int compressionPower = 1;
        DNDS_DECLARE_CONFIG(ImplicitPositivitySettings)
        {
            DNDS_FIELD(rhsMode, "none, rhs compression, or dTau compression; active only with numerics.positivity");
            DNDS_FIELD(rhsReduction, "cell, minimumLimited or minimumAll; Euler global alpha flags, before RHS/dTau application");
            DNDS_FIELD(rhsScale, "Euler RHS-PP trial increment multiplier");
            DNDS_FIELD(rhsRelax, "Euler RHS/dTau density and energy relaxation in (0,1]");
            DNDS_FIELD(incrementRelax, "Euler accepted-correction relaxation in (0,1]");
            DNDS_FIELD(compressionPower, "Positive integer exponent multiplier in Euler alpha compression");
        }
    };
    struct PolynomialMultigridSettings
    {
        int levels = 0;
        index cycles = 50, preSmooth = 1, postSmooth = 1, coarseSmooth = 4;
        real damping = 1;
        std::vector<index> updateComponents;
        DNDS_DECLARE_CONFIG(PolynomialMultigridSettings)
        {
            DNDS_FIELD(levels, "0 disables pMG; 1 adds direct-gradient O2; 2 also adds first-order convection, retaining viscosity");
            DNDS_FIELD(cycles, "Maximum FAS V-cycles per physical stage");
            DNDS_FIELD(preSmooth, "Bounded nonlinear corrections before restriction");
            DNDS_FIELD(postSmooth, "Bounded nonlinear corrections after prolongation");
            DNDS_FIELD(coarseSmooth, "Bounded nonlinear corrections at the coarsest level");
            DNDS_FIELD(damping, "Physical coarse-correction multiplier in (0,1]");
            DNDS_FIELD(updateComponents, "Distinct flow5 indices updated by pMG; empty updates all. Other means remain frozen; finest convergence still uses residualComponents");
        }
    };
    struct ImplicitSettings
    {
        std::string diagonal = "entries", jacobianAction = "assembled";
        index iterations = 100, minIterations = 1;
        index consoleInterval = 0;
        real relativeTolerance = 1e-8, absoluteTolerance = 1e-11;
        int normOrder = 2;
        std::string residualReference = "initial";
        std::vector<int> residualComponents{0, 1, 2, 3, 4};
        real pseudoCFL = 10, maxPseudoCFL = 1e4, pseudoGrowth = 1.3, relaxation = 1;
        bool roeDissipation = false, collectiveChecks = true;
        bool requireConvergedReconstruction = false;
        bool rejectUnconvergedStages = false;
        ImplicitLinearSettings linear;
        ImplicitSGSSettings sgs;
        ImplicitILUSettings ilu;
        ImplicitPositivitySettings positivity;
        PolynomialMultigridSettings multigrid;
        void Validate() const;
        DNDS_DECLARE_CONFIG(ImplicitSettings)
        {
            DNDS_FIELD(diagonal, "entries or full block diagonal storage");
            DNDS_FIELD(jacobianAction, "assembled or semiMatrixFree off-diagonal blocks; the diagonal and local ILU factors remain explicit");
            DNDS_FIELD(iterations, "Maximum nonlinear corrections per physical stage");
            DNDS_FIELD(minIterations, "Minimum corrections before accepting a stage");
            DNDS_FIELD(consoleInterval, "Rank-zero nonlinear residual/VR progress interval; zero disables it");
            DNDS_FIELD(relativeTolerance, "Selected-component stage residual tolerance relative to its configured reference");
            DNDS_FIELD(absoluteTolerance, "Per-component absolute volume-weighted stage residual tolerance");
            DNDS_FIELD(normOrder, "Volume-weighted residual norm: 1 or 2");
            DNDS_FIELD(residualReference, "initial or runningMaximum per-component reference within one solve");
            DNDS_FIELD(residualComponents, "Distinct flow5 component indices required for convergence; all five by default");
            DNDS_FIELD(pseudoCFL, "Initial local pseudo-time CFL for each stage");
            DNDS_FIELD(maxPseudoCFL, "Maximum pseudo-time CFL");
            DNDS_FIELD(pseudoGrowth, "CFL growth factor after each nonlinear correction");
            DNDS_FIELD(relaxation, "Multiplier on each computed correction in (0,1]");
            DNDS_FIELD(roeDissipation, "Use full equation-owned characteristic dissipation; requires block diagonal");
            DNDS_FIELD(collectiveChecks, "Coordinate local failures between implicit kernels; false selects caller-ordered eager execution");
            DNDS_FIELD(requireConvergedReconstruction, "Require the requested VR tolerance as well as the mean equation target; false preserves Euler-style fixed-sweep acceptance");
            DNDS_FIELD(rejectUnconvergedStages, "Reject a finite admissible stage or DITR outer solve at its iteration cap; false advances the final iterate like Euler while retaining an unconverged report");
            config.field_section(&T::linear, "linear", "Linear solve controls");
            config.field_section(&T::sgs, "sgs", "Symmetric relaxation controls");
            config.field_section(&T::ilu, "ilu", "Rank-local factorization controls");
            config.field_section(&T::positivity, "positivity", "Euler-derived implicit PP controls");
            config.field_section(&T::multigrid, "multigrid", "Same-cell polynomial FAS hierarchy");
        }
    };
    struct DomainSettings
    {
        EquationSettings equation;
        ExecutionSettings execution;
        MeshSettings mesh;
        RigidMotionSettings motion;
        NumericsSettings numerics;
        ImplicitSettings implicit;
        std::array<real, 5> initialPrimitive{1, 0, 0, 0, 1};
        FlowProfileSettings initialProfile;
        std::map<std::string, BoundarySettings> boundaries;
        DNDS_DECLARE_CONFIG(DomainSettings)
        {
            config.field_section(&T::equation, "equation", "Equation and constitutive laws");
            config.field_section(&T::execution, "execution", "Explicit execution capabilities");
            config.field_section(&T::mesh, "mesh", "Mesh input");
            config.field_section(&T::motion, "motion", "Prescribed rigid frame and mesh velocity; absolute flow state in mesh axes");
            config.field_section(&T::numerics, "numerics", "Discrete operators");
            config.field_section(&T::implicit, "implicit", "Nonlinear CPU implicit stage controls");
            DNDS_FIELD(initialPrimitive, "Constant initial rho,u,v,w,p");
            config.field_section(&T::initialProfile, "initialProfile", "Quadrature-averaged initial physical field");
            config.field_map_of<BoundarySettings>(&T::boundaries, "boundaries", "Boundary-zone ID to condition");
        }
    };
    /// Cauchy's ordered reconstruction uses one logical core-first order under
    /// both overlap policies. Standalone CFV retains natural-order All support.
    /// Resolve before publishing root config and when constructing FV directly.
    DomainSettings ResolveSweepOrdering(DomainSettings settings);
    struct DITRTimeSettings
    {
        real fraction = .55, theta = .9146;
        index iterations = 100, minIterations = 0;
        real relativeTolerance = 1e-8, absoluteTolerance = 1e-10;
        void Validate() const;
        DNDS_DECLARE_CONFIG(DITRTimeSettings)
        {
            DNDS_FIELD(fraction, "U2R2 intermediate time fraction; CPU positive-scale profile requires (0,2/3)");
            DNDS_FIELD(theta, "Positive H+theta*I intermediate-equation weight");
            DNDS_FIELD(iterations, "Maximum coupled outer corrections; implicit.rejectUnconvergedStages selects rejection at the cap");
            DNDS_FIELD(minIterations, "Minimum coupled corrections before early convergence of both original equations");
            DNDS_FIELD(relativeTolerance, "Per-component outer H/I norm target relative to each initial norm");
            DNDS_FIELD(absoluteTolerance, "Per-component absolute outer H/I norm target");
        }
    };
    struct PhysicalTimePositivitySettings
    {
        bool enabled = false;
        real relax = .8, scale = 1, increaseLimit = 2;
        void Validate() const;
        DNDS_DECLARE_CONFIG(PhysicalTimePositivitySettings)
        {
            DNDS_FIELD(enabled, "Euler physical dt PP bound from a zero-reconstruction trial before time-equation preparation");
            DNDS_FIELD(relax, "Euler density/internal-energy relaxation in (0,1]");
            DNDS_FIELD(scale, "Positive trial-increment multiplier dtPPLimitScale");
            DNDS_FIELD(increaseLimit, "Maximum timestep growth from the last accepted positive step; at least one");
        }
    };
    struct TimeSettings
    {
        std::string method = "ssprk3";
        bool steady = false;
        real dt = 1e-3;
        index steps = 1;
        real finalTime = 0, cfl = 0, wallSeconds = 0;
        real steadyTolerance = 0;
        index steadyMinSteps = 100;
        int maxRetries = 0;
        DITRTimeSettings ditr;
        PhysicalTimePositivitySettings positivity;
        DNDS_DECLARE_CONFIG(TimeSettings)
        {
            DNDS_FIELD(method, "Coordinated physical-time method");
            DNDS_FIELD(steady, "Euler steady override: BDF1 infinite-dt limit, one outer solve and frozen physical time");
            DNDS_FIELD(dt, "Fixed positive physical timestep");
            DNDS_FIELD(steps, "Nonnegative accepted step target");
            DNDS_FIELD(finalTime, "Positive physical-time target; zero uses the step target");
            DNDS_FIELD(cfl, "Positive CFL enables adaptive global stepping, capped by dt");
            DNDS_FIELD(wallSeconds, "Collective graceful wall stop; zero disables it");
            DNDS_FIELD(maxRetries, "Maximum timestep halvings after a rejected candidate");
            DNDS_FIELD(steadyTolerance, "Relative accepted-state residual target; zero means physical transient");
            DNDS_FIELD(steadyMinSteps, "Minimum accepted steps before a steady stop");
            config.field_section(&T::ditr, "ditr", "Coupled U2R2 time equations");
            config.field_section(&T::positivity, "positivity", "Euler physical timestep positivity/growth policy");
        }
    };
    struct OutputSettings
    {
        std::string prefix = "cauchy";
        index interval = 1;
        index historyInterval = 1;
        int hdfDeflateLevel = 0;
        bool checkpointReconstruction = false;
        std::vector<int> forceZones;
        DNDS_DECLARE_CONFIG(OutputSettings)
        {
            DNDS_FIELD(prefix, "Output file prefix");
            DNDS_FIELD(interval, "Accepted-step snapshot interval; zero disables periodic output");
            DNDS_FIELD(historyInterval, "CSV diagnostics cadence in accepted steps");
            DNDS_FIELD(hdfDeflateLevel, "VTKHDF compression level 0..9; uncompressed is the initial portable baseline");
            DNDS_FIELD(checkpointReconstruction, "Store optional compatible polynomial warm guesses for exact finite-sweep continuation; default omits them");
            DNDS_FIELD(forceZones, "Boundary zones on which pressure and viscous body force are integrated");
        }
    };
    struct DomainConfig
    {
        DomainRankGroup placement;
        DomainSettings settings;
    };
    /// Directed point-interpolation connection. Configure two entries for a
    /// two-way interface. Zone IDs are scoped to their respective domains.
    struct InterfaceSettings
    {
        std::string donor, receiver, routing = "coarseBoxes";
        int donorZone = 0, receiverZone = 0, coarseBoxesPerRank = 4;
        real maxGap = 1e-8;
        DNDS_DECLARE_CONFIG(InterfaceSettings)
        {
            DNDS_FIELD(donor, "Donor domain name");
            DNDS_FIELD(receiver, "Receiver domain name");
            DNDS_FIELD(donorZone, "Selected donor boundary zone");
            DNDS_FIELD(receiverZone, "Interface receiver boundary zone");
            DNDS_FIELD(routing, "coarseBoxes or allGatherBoundary correctness baseline");
            DNDS_FIELD(coarseBoxesPerRank, "Maximum coarse routing boxes per donor rank");
            DNDS_FIELD(maxGap, "Maximum physical point-to-surface projection distance");
        }
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
            std::vector<InterfaceSettings> interfaces;
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
        const std::vector<InterfaceSettings> &Interfaces() const { return _data->interfaces; }
        std::vector<DomainRankGroup> RankGroups() const;
    };
}
