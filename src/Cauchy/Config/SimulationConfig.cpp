#include "Cauchy/Config/SimulationConfig.hpp"

#include "Cauchy/Physics/StateInput.hpp"
#include "DNDS/Config/JsonSchema.hpp"
#include "DNDS/MPI/Failure.hpp"
#include <set>
#include <numeric>
#include <optional>

namespace DNDS::Cauchy
{
    namespace
    {
        std::string Escape(std::string text)
        {
            std::string result;
            for (char c : text)
                result += c == '~' ? "~0" : c == '/' ? "~1"
                                                     : std::string(1, c);
            return result;
        }
        void Allowed(const Json &value, std::initializer_list<const char *> allowed, const std::string &path)
        {
            DNDS_check_throw_info(value.is_object(), "Cauchy configuration object required at " + path);
            for (const auto &[key, child] : value.items())
                DNDS_check_throw_info(std::find(allowed.begin(), allowed.end(), key) != allowed.end(), "unknown Cauchy key " + path + "/" + key);
        }
        using Origins = std::map<std::string, std::string>;
        void Origin(const Json &value, const std::string &path, const std::string &source, Origins &origins)
        {
            if ((value.is_object() || value.is_array()) && !value.empty())
                for (const auto &[key, child] : value.items())
                    Origin(child, path + "/" + Escape(key), source, origins);
            else
                origins[path] = source;
        }
        void Merge(Json &target, const Json &patch, const std::string &path, const std::string &source, Origins &origins)
        {
            if (patch.is_object() && target.is_object())
            {
                const std::string limitedVRPath = "/numerics/limitedVR";
                if (path.size() >= limitedVRPath.size() && path.compare(path.size() - limitedVRPath.size(), limitedVRPath.size(), limitedVRPath) == 0 && patch.contains("variant"))
                {
                    // Like an input preset, a variant selects explicit defaults
                    // for this layer; fields beside it then override them and a
                    // later layer wins. Save expanded controls and provenance.
                    target = CFV::LimitedVRSettings::ForVariant(patch.at("variant").get<std::string>());
                    Origin(target, path, source + " LVR variant defaults", origins);
                }
                if (!patch.empty())
                    origins.erase(path);
                for (const auto &[key, child] : patch.items())
                {
                    if (!target.contains(key))
                        target[key] = child.is_object() ? Json::object() : Json();
                    Merge(target[key], child, path + "/" + Escape(key), source, origins);
                }
            }
            else
            {
                for (auto it = origins.begin(); it != origins.end();)
                    if (it->first == path || it->first.compare(0, path.size() + 1, path + "/") == 0)
                        it = origins.erase(it);
                    else
                        ++it;
                target = patch;
                Origin(target, path, source, origins);
            }
        }
        void PrimitiveValid(const std::array<real, 5> &values)
        {
            DNDS_check_throw_info(std::all_of(values.begin(), values.end(), [](real value)
                                              { return std::isfinite(value); }) &&
                                      values[0] > 0 && values[4] > 0,
                                  "Cauchy primitive input needs finite values and positive rho,p");
        }
        void Validate(const DomainConfig &domain, int size)
        {
            const auto &settings = domain.settings;
            settings.implicit.ValidateComponents(settings.equation.eos == "perfectGas" ? 5 : INT_MAX - 1);
            settings.motion.Validate(settings.mesh.dimension);
            DNDS_check_throw_info(!settings.implicit.requireConvergedReconstruction || settings.numerics.reconstruction != "variational" || settings.numerics.reconstructionTolerance > 0,
                                  "strict implicit VR acceptance requires a positive reconstructionTolerance");
            DNDS_check_throw_info(!domain.placement.name.empty() && !domain.placement.ranks.empty(), "Cauchy domain needs a name and ranks");
            std::set<int> ranks;
            for (int rank : domain.placement.ranks)
                DNDS_check_throw_info(rank >= 0 && rank < size && ranks.insert(rank).second, "invalid/duplicate Cauchy domain rank");
            const auto &equation = settings.equation;
            ResolveReferenceScales(equation);
            DNDS_check_throw_info(std::isfinite(equation.gamma) && equation.gamma > 1 && std::isfinite(equation.gasConstant) && equation.gasConstant > 0 &&
                                      std::isfinite(equation.viscosity) && equation.viscosity >= 0 && std::isfinite(equation.prandtl) && equation.prandtl > 0,
                                  "invalid Cauchy constitutive constants");
            DNDS_check_throw_info(equation.transport != "inviscid" || equation.viscosity == 0, "inviscid configuration has unused nonzero viscosity");
            if (equation.eos == "perfectGas")
                EquationSystem::Resolve(equation);
            DNDS_check_throw_info(settings.mesh.dimension == 2 || settings.mesh.dimension == 3, "Cauchy geometric dimension must be 2 or 3");
            DNDS_check_throw_info(settings.execution.hostThreads >= 0 && settings.execution.hostThreads <= 1024, "invalid CPU worker count");
            const auto &numerics = settings.numerics;
            numerics.ValidatePositivity();
            DNDS_check_throw_info(numerics.reconstructionIterations > 0 && std::isfinite(numerics.reconstructionTolerance) && numerics.reconstructionTolerance >= 0,
                                  "invalid Cauchy numerical controls");
            numerics.cfv.Validate();
            numerics.limitedVR.Validate();
            const auto &riemann = numerics.riemann;
            DNDS_check_throw_info((numerics.flux != "Roe_M3" && numerics.flux != "Roe_M5") || riemann.fixScale > 0,
                                  "low-dissipation Roe flux requires a positive fixScale divisor");
            DNDS_check_throw_info(std::isfinite(riemann.fixScale) && riemann.fixScale >= 0 && std::isfinite(riemann.incrementScale) && riemann.incrementScale >= 0 &&
                                      std::isfinite(riemann.dLambda) && riemann.dLambda >= 0 && std::isfinite(riemann.referenceMach) && riemann.referenceMach > 0 &&
                                      std::isfinite(riemann.pressureStencilMin) && riemann.pressureStencilMin > 0,
                                  "invalid Cauchy Riemann controls");
            PrimitiveValid(settings.initialPrimitive);
            DNDS_check_throw_info(std::isfinite(settings.initialTemperature) && settings.initialTemperature >= 0 &&
                                      (equation.eos != "perfectGas" || settings.initialMassFractions.empty()),
                                  "invalid constant initial thermodynamic input");
            const auto validateProfile = [](const FlowProfileSettings &profile)
            {
                DNDS_check_throw_info(profile.kind == "constant" || profile.kind == "isentropicVortex" || profile.kind == "doubleMachReflection" || profile.kind == "cylinderWake", "unknown physical field profile");
                DNDS_check_throw_info(std::isfinite(profile.period) && profile.period > 0 && std::isfinite(profile.strength) && std::isfinite(profile.perturbation) &&
                                          std::all_of(profile.center.begin(), profile.center.end(), [](real value)
                                                      { return std::isfinite(value); }),
                                      "invalid field profile parameters");
            };
            validateProfile(settings.initialProfile);
            for (auto length : settings.mesh.periodicLengths)
                DNDS_check_throw_info(std::isfinite(length) && length >= 0, "periodic lengths must be finite and nonnegative");
            for (const auto &[name, zone] : settings.mesh.boundaryIds)
                DNDS_check_throw_info(!name.empty() && zone != 0 && zone != 1 && zone >= -6,
                                      "mesh.boundaryIds reserves 0 for interior, 1 for null, and -1..-6 for periodic boundaries");
            for (const auto &[zone, condition] : settings.boundaries)
            {
                size_t parsed = 0;
                const auto id = std::stoll(zone, &parsed);
                DNDS_check_throw_info(parsed == zone.size() && id >= INT_MIN && id <= INT_MAX && std::to_string(id) == zone,
                                      "boundary key must be a canonical integer zone ID");
                PrimitiveValid(condition.primitive);
                DNDS_check_throw_info(condition.massFractions.empty() ||
                                          (equation.eos != "perfectGas" && (condition.kind == "farfield" || condition.kind == "analytic")),
                                      "boundary mass fractions require a prescribed mixture state");
                validateProfile(condition.profile);
                if (condition.kind == "noSlipWall" && condition.thermal == "isothermal")
                    DNDS_check_throw_info(condition.temperature >= numerics.temperatureFloor, "wall temperature lies below the configured temperature floor");
                if (condition.kind == "farfield" && equation.eos == "perfectGas" && condition.state.is_null())
                    DNDS_check_throw_info(condition.primitive[4] / (condition.primitive[0] * equation.gasConstant) >= numerics.temperatureFloor,
                                          "farfield temperature lies below the configured temperature floor");
                DNDS_check_throw_info(std::isfinite(condition.temperature) && condition.temperature > 0 &&
                                          std::all_of(condition.velocity.begin(), condition.velocity.end(), [](real v)
                                                      { return std::isfinite(v); }),
                                      "invalid boundary state");
                DNDS_check_throw_info(condition.kind == "noSlipWall" || (condition.thermal == "adiabatic" && (condition.kind == "slipWall" || condition.velocity == std::array<real, 3>{0, 0, 0})),
                                      "wall thermal/velocity parameters supplied to a non-wall condition");
            }
        }
    }
    void NumericsSettings::ValidatePositivity() const
    {
        DNDS_check_throw_info(std::isfinite(temperatureFloor) && temperatureFloor > 0 && std::isfinite(densityFloor) && densityFloor > 0 &&
                                  std::isfinite(pressureFloor) && pressureFloor > 0 && std::isfinite(sensibleEnergyFloor) && sensibleEnergyFloor >= 0 &&
                                  std::isfinite(positivityBetaCutoff) && positivityBetaCutoff >= 0 && positivityBetaCutoff < 1 &&
                                  (positivityReconstruction == "Euler" || positivityReconstruction == "Segment") &&
                                  (positivityClippedReferencePolicy == "fallbackToBase" || positivityClippedReferencePolicy == "backtracking") &&
                                  positivityBetaPower >= 0 && positivityBetaPower < INT_MAX,
                              "invalid reconstruction positivity controls");
        DNDS_check_throw_info(!positivity || positivityReconstruction != "Segment" || (positivityCompressToMean && positivityBetaPower == 1),
                              "Segment compatibility does not implement Euler order-reference or beta-power controls");
    }
    void DITRTimeSettings::Validate() const
    {
        DNDS_check_throw_info((variant == "U2R2" || variant == "U2R1") && std::isfinite(fraction) && fraction > 0 && fraction < real(2) / 3 &&
                                  std::isfinite(theta) && theta > 0 && (!iterations || *iterations > 0) && (!minIterations || *minIterations >= 0) &&
                                  (!iterations || !minIterations || *minIterations <= *iterations) &&
                                  (!relativeTolerance || (std::isfinite(*relativeTolerance) && *relativeTolerance >= 0)) &&
                                  (!absoluteTolerance || (std::isfinite(*absoluteTolerance) && *absoluteTolerance >= 0)),
                              "invalid/unavailable DITR positive-stage-scale controls");
    }
    ImplicitSettings DITRTimeSettings::ResolveControls(const ImplicitSettings &common) const
    {
        Validate();
        auto result = common;
        if (iterations)
            result.iterations = *iterations;
        if (minIterations)
            result.minIterations = *minIterations;
        if (relativeTolerance)
            result.relativeTolerance = *relativeTolerance;
        if (absoluteTolerance)
            result.absoluteTolerance = *absoluteTolerance;
        DNDS_check_throw_info(result.iterations > 0 && result.minIterations >= 0 && result.minIterations <= result.iterations &&
                                  std::isfinite(result.relativeTolerance) && result.relativeTolerance >= 0 &&
                                  std::isfinite(result.absoluteTolerance) && result.absoluteTolerance >= 0,
                              "DITR overrides conflict with the common implicit iteration/tolerance controls");
        return result;
    }
    void PhysicalTimePositivitySettings::Validate() const
    {
        DNDS_check_throw_info(std::isfinite(relax) && relax > 0 && relax <= 1 && std::isfinite(scale) && scale > 0 &&
                                  std::isfinite(increaseLimit) && increaseLimit >= 1,
                              "invalid Euler physical timestep positivity/growth controls");
    }
    void ImplicitSettings::Validate() const
    {
        ValidateComponents(5);
    }
    void ImplicitSettings::ValidateComponents(index components) const
    {
        DNDS_check_throw_info(components > 0 && components < INT_MAX, "invalid implicit equation dimension");
        DNDS_check_throw_info((normOrder == 1 || normOrder == 2) &&
                                  (residualReference == "initial" || residualReference == "runningMaximum"),
                              "invalid implicit residual norm/reference/components");
        std::vector<std::string> selected;
        for (const auto &component : residualComponents)
        {
            DNDS_check_throw_info(!component.empty() && std::find(selected.begin(), selected.end(), component) == selected.end(), "implicit residual components must be distinct nonempty names");
            selected.push_back(component);
        }
        const auto nonnegative = [](real value)
        { return std::isfinite(value) && value >= 0; };
        std::vector<int> updated;
        for (const auto component : multigrid.updateComponents)
        {
            DNDS_check_throw_info(component >= 0 && component < components && std::find(updated.begin(), updated.end(), component) == updated.end(), "pMG update components must be distinct equation indices");
            updated.push_back(component);
        }
        DNDS_check_throw_info((diagonal == "entries" || diagonal == "block") && (!roeDissipation || diagonal == "block") &&
                                  (jacobianAction == "assembled" || jacobianAction == "directEvaluated" || jacobianAction == "semiMatrixFree") &&
                                  !updatePolicy.empty() &&
                                  iterations > 0 && minIterations >= 0 && minIterations <= iterations && consoleInterval >= 0 &&
                                  nonnegative(relativeTolerance) && nonnegative(absoluteTolerance) &&
                                  std::isfinite(pseudoCFL) && pseudoCFL > 0 && std::isfinite(maxPseudoCFL) && maxPseudoCFL >= pseudoCFL &&
                                  std::isfinite(pseudoGrowth) && pseudoGrowth >= 1 && std::isfinite(relaxation) && relaxation > 0 && relaxation <= 1,
                              "invalid implicit nonlinear/diagonal/pseudo-time controls");
        DNDS_check_throw_info((linear.method == "sgs" || linear.method == "jacobi" || linear.method == "ilu" || linear.method == "gmres") && (linear.preconditioner == "sgs" || linear.preconditioner == "ilu") &&
                                  linear.jacobiLocalSweeps >= 1 && linear.jacobiLocalSweeps <= 2 &&
                                  (linear.componentScaling == "none" || linear.componentScaling == "meanState") &&
                                  (linear.stopping == "residual" || (linear.stopping == "fixedCount" && linear.method != "gmres")) &&
                                  linear.iterations > 0 && linear.subspace > 0 && linear.subspace < INT_MAX &&
                                  nonnegative(linear.relativeTolerance) && nonnegative(linear.absoluteTolerance) && sgs.sweeps > 0 && ilu.fill >= -1 && ilu.fill <= 5 && ilu.preconditionerCorrections > 0 &&
                                  (ilu.ordering == "natural" || ilu.ordering == "rcm") &&
                                  !((linear.method == "ilu" || linear.method == "jacobi" || (linear.method == "gmres" && linear.preconditioner == "ilu")) && sgs.exchangeBetweenSweeps),
                              "invalid implicit linear/SGS/ILU controls");
        DNDS_check_throw_info((positivity.rhsFPPMode == "none" || positivity.rhsFPPMode == "rhs" || positivity.rhsFPPMode == "dTau") &&
                                  (positivity.rhsFPPReduction == "cell" || positivity.rhsFPPReduction == "minimumLimited" || positivity.rhsFPPReduction == "minimumAll") &&
                                  nonnegative(positivity.rhsFPPScale) && std::isfinite(positivity.rhsFPPRelax) && positivity.rhsFPPRelax > 0 && positivity.rhsFPPRelax <= 1 &&
                                  std::isfinite(positivity.incrementRelax) && positivity.incrementRelax > 0 && positivity.incrementRelax <= 1 &&
                                  positivity.compressionPower > 0 && positivity.compressionPower <= INT_MAX / 2,
                              "invalid implicit Euler-PP controls");
        DNDS_check_throw_info(multigrid.levels >= 0 && multigrid.levels <= 2 && multigrid.cycles > 0 &&
                                  multigrid.preSmooth >= 0 && multigrid.postSmooth >= 0 && multigrid.coarseSmooth > 0 &&
                                  (!multigrid.levels || (multigrid.preSmooth <= iterations && multigrid.postSmooth <= iterations && multigrid.coarseSmooth <= iterations)) &&
                                  std::isfinite(multigrid.damping) && multigrid.damping > 0 && multigrid.damping <= 1,
                              "invalid polynomial FAS controls");
        for (const auto &[name, level] : multigrid.coarseLevelSettings)
        {
            DNDS_check_throw_info((name == "1" || name == "2") && level.preSmooth >= -1 && level.postSmooth >= -1 &&
                                      level.preSmooth <= iterations && level.postSmooth <= iterations,
                                  "invalid coarse level key or smoothing count");
            const index number = name == "1" ? 1 : 2;
            if (number == multigrid.levels)
            {
                const index pre = level.preSmooth < 0 ? multigrid.coarseSmooth : level.preSmooth;
                const index post = level.postSmooth < 0 ? 0 : level.postSmooth;
                DNDS_check_throw_info(pre <= iterations - post && pre + post > 0, "invalid bottom-level correction count");
            }
            ResolveCoarseLevel(number);
        }
    }
    ImplicitSettings ImplicitSettings::ResolveCoarseLevel(index level) const
    {
        DNDS_check_throw_info(level == 1 || level == 2, "coarse smoother level must be 1 or 2");
        auto result = *this;
        result.multigrid.levels = 0;
        result.multigrid.coarseLevelSettings.clear();
        const auto found = multigrid.coarseLevelSettings.find(std::to_string(level));
        if (found != multigrid.coarseLevelSettings.end())
        {
            Config::ValidatePartialJsonSchema(found->second.solver, MultigridSmootherSchema(),
                                              "/implicit/multigrid/coarseLevelSettings/" + std::to_string(level) + "/solver");
            Json merged = result;
            merged.merge_patch(found->second.solver);
            result = merged.get<ImplicitSettings>();
        }
        result.Validate();
        return result;
    }
    DomainSettings ResolveSweepOrdering(DomainSettings settings)
    {
        if (settings.numerics.cfv.SORInstead)
            settings.numerics.cfv.coreFirst = true;
        return settings;
    }
    void ExecutionSettings::ValidateMPIThreadSupport(int provided) const
    {
        DNDS_check_throw_info(hostThreads == 0 || provided >= MPI_THREAD_FUNNELED,
                              "Cauchy CPU workers require at least MPI_THREAD_FUNNELED; disable workers or request a higher MPI thread level");
    }
    ResolvedExecutionPlan ResolvedExecutionPlan::Resolve(const DomainConfig &domain)
    {
        const auto &s = domain.settings;
        std::vector<std::string> missing;
        auto require = [&](bool found, const std::string &operation)
        { if (!found) missing.push_back(operation); };
        require(s.execution.device == "cpu", "residual/BC/reconstruction backend " + s.execution.device);
        require(s.execution.layout == "aos" || s.execution.layout == "soa" || s.execution.layout == "aosoa32", "flow layout " + s.execution.layout);
        require(s.execution.communication == "mpi-direct", "transport " + s.execution.communication);
        require(s.execution.communicationExecution == "caller" || s.execution.communicationExecution == "worker", "communication execution " + s.execution.communicationExecution);
        if (s.execution.communicationExecution == "worker")
        {
            int provided = MPI_THREAD_SINGLE;
            MPI_Query_thread(&provided);
            require(provided == MPI_THREAD_MULTIPLE, "communication worker requires MPI_THREAD_MULTIPLE");
        }
        require(s.execution.transferPreflight == "collective" || s.execution.transferPreflight == "abortOnFailure", "transfer preflight " + s.execution.transferPreflight);
        require(s.execution.fieldGrouping == "equationBlocks" || s.execution.fieldGrouping == "coupled", "field grouping " + s.execution.fieldGrouping);
        require(s.equation.system == "compressible", "equation " + s.equation.system);
        const bool mixture = s.equation.eos == "idealGasMixture";
        const bool rans = s.equation.rans.model != "none", extended = mixture || rans;
        require(rans || s.initialRANS.empty(), "initialRANS without a RANS model");
        if (rans)
        {
            require((s.equation.rans.model == "SA" || s.equation.rans.model == "KOmegaWilcox" ||
                     s.equation.rans.model == "KOmegaSST" || s.equation.rans.model == "RealizableKE"),
                    "RANS spatial profile " + s.equation.rans.model);
            require(s.equation.transport == "NewtonianFourier", "RANS requires viscous transport");
        }
        require(s.equation.eos == "perfectGas" || mixture, "EOS " + s.equation.eos);
#ifndef DNDS_USE_CANTERA
        require(!mixture, "idealGasMixture requires Cantera in this build");
#endif
        require(s.equation.transport == "inviscid" || s.equation.transport == "NewtonianFourier", "constitutive law " + s.equation.transport);
        require(s.equation.source == "none" || (mixture && s.equation.source == "chemical"), "volume source " + s.equation.source);
        if (mixture)
        {
            const auto &material = s.equation.mixture;
            require(!material.mechanismFile.empty(), "mixture mechanismFile");
            require(std::isfinite(material.rho0) && material.rho0 > 0 && std::isfinite(material.U0) && material.U0 > 0 &&
                        std::isfinite(material.L0) && material.L0 > 0 && std::isfinite(material.T0) && material.T0 > 0,
                    "mixture reference scales");
            require(s.equation.viscosity == 0 && s.equation.viscosityModel == "constant" &&
                        s.equation.viscosityReferenceTemperature == 1 && s.equation.sutherlandConstant == 0,
                    "mixture transport uses its material, not a perfect-gas viscosity law");
        }
        if (extended)
        {
            require(s.initialProfile.kind == "constant", "nonconstant extended equation initialization");
            const auto flux = Riemann::RiemannFlux::Resolve(s.numerics.flux);
            require((!mixture || flux.Kind() == Riemann::Solver::Rusanov) && !s.numerics.faceFallback,
                    "extended equation thermodynamic flux preparation/face fallback");
            require(!s.numerics.riemann.useStencil || (!flux.UsesTransverseSignal() && !flux.UsesPressureStencil()),
                    "extended equation wave/pressure stencil auxiliary inputs");
            require(s.numerics.limiter == "none" || s.numerics.limiter == "Barth" ||
                        (s.numerics.limiter == "WBAP" && (!s.numerics.characteristicLimiter || !mixture)),
                    "extended equation limiter");
            require(s.equation.transport != "NewtonianFourier" || s.numerics.viscousGradient == "polynomial", "extended equation viscous gradient");
            require(!s.implicit.roeDissipation, "extended equation Roe Jacobian");
        }
        require(s.initialField.Empty() || s.initialProfile.kind == "constant", "spatial overrides require a constant background profile");
        require(Riemann::RiemannFlux::Available(s.numerics.flux), "numerical flux " + s.numerics.flux);
        require(s.numerics.reconstruction == "constant" || s.numerics.reconstruction == "gradient" || s.numerics.reconstruction == "variational", "reconstruction " + s.numerics.reconstruction);
        require(s.implicit.sourceJacobianMode == "center" || s.implicit.sourceJacobianMode == "rhsQuadrature", "source Jacobian mode " + s.implicit.sourceJacobianMode);
        require(s.implicit.sourceJacobianMode != "rhsQuadrature" || s.numerics.reconstruction == "variational", "rhsQuadrature source Jacobian requires variational reconstruction");
        require(s.implicit.sourceJacobianMode != "rhsQuadrature" || s.equation.source != "none" || rans, "rhsQuadrature source Jacobian requires an active volume source");
        require(s.numerics.gradient == "leastSquares" || s.numerics.gradient == "greenGauss" || s.numerics.gradient == "ggmp", "gradient " + s.numerics.gradient);
        require(s.numerics.viscousGradient == "primitive" || s.numerics.viscousGradient == "polynomial", "viscous gradient " + s.numerics.viscousGradient);
        require(s.numerics.limiter == "none" || s.numerics.limiter == "Barth" ||
                    s.numerics.limiter == "WBAP" || s.numerics.limiter == "LVR",
                "limiter " + s.numerics.limiter);
        require(s.numerics.limiter != "Barth" || s.numerics.reconstruction == "gradient",
                "Barth needs direct gradient reconstruction");
        require(s.numerics.limiter != "WBAP" || s.numerics.reconstruction != "constant",
                "WBAP needs gradient or variational reconstruction");
        require(s.numerics.limiter != "LVR" || (s.numerics.reconstruction == "variational" && s.numerics.cfv.maxOrder >= 2), "LVR needs degree2/3 variational reconstruction");
        require(!s.implicit.multigrid.levels || (s.numerics.reconstruction == "variational" && s.numerics.cfv.maxOrder >= 2), "polynomial FAS requires high-order variational reconstruction");
        for (const auto &[zone, condition] : s.boundaries)
        {
            require(condition.kind == "extrapolate" || condition.kind == "farfield" || condition.kind == "slipWall" || condition.kind == "noSlipWall" || condition.kind == "analytic" || condition.kind == "doubleMachBottom" || condition.kind == "interface", "BC zone " + zone + ": " + condition.kind);
            require(condition.kind != "doubleMachBottom" || condition.profile.kind == "doubleMachReflection", "DMR bottom requires its analytic shock profile");
            require(condition.thermal == "adiabatic" || condition.thermal == "isothermal", "thermal BC " + condition.thermal);
            require(condition.wallVelocityFrame == "mesh" || condition.wallVelocityFrame == "ground", "wall velocity frame " + condition.wallVelocityFrame);
            require(!extended || (condition.kind != "doubleMachBottom" &&
                                  (condition.kind != "analytic" ||
                                   condition.profile.kind == "constant")),
                    "extended equation boundary profile " + condition.kind);
            require(condition.kind != "noSlipWall" || s.equation.transport == "NewtonianFourier", "no-slip wall needs viscous transport");
        }
        if (!missing.empty())
        {
            std::string message = "Cauchy domain " + domain.placement.name + " has missing capabilities:";
            for (const auto &item : missing)
                message += "\n  " + item;
            DNDS_check_throw_info(false, message);
        }
        return {{std::string(mixture ? "cpu/idealGasMixture/flow5+species" : "cpu/perfectGas/flow5") + (rans ? "+" + s.equation.rans.model : ""), "cpu/" + s.numerics.flux, "cpu/" + s.equation.transport,
                 "cfv/" + s.numerics.reconstruction, "cfv/" + s.numerics.limiter, "runtime/mpi-direct"}};
    }
    Json SimulationConfig::InputSchema()
    {
        const Json domain{{"type", "object"},
                          {"additionalProperties", false},
                          {"required", {"name"}},
                          {"properties",
                           {{"name", {{"type", "string"}, {"description", "Unique domain name"}}},
                            {"ranks", {{"type", "array"}, {"items", {{"type", "integer"}}}, {"minItems", 1}}},
                            {"profiles", {{"type", "array"}, {"items", {{"type", "string"}}}}},
                            {"overrides", DomainSettings::schema("Per-domain settings patch")}}}};
        Json schema{{"$schema", "http://json-schema.org/draft-07/schema#"},
                    {"title", "DNDSR Cauchy simulation input"},
                    {"type", "object"},
                    {"additionalProperties", false},
                    {"properties",
                     {{"schemaVersion", {{"type", "integer"}, {"enum", {1}}, {"default", 1}}},
                      {"profiles", {{"type", "object"}, {"additionalProperties", DomainSettings::schema("Named settings patch")}}},
                      {"defaults", DomainSettings::schema("Root domain-settings patch")},
                      {"domains", {{"type", "array"}, {"items", domain}, {"minItems", 1}}},
                      {"time", TimeSettings::schema("Physical-time settings patch")},
                      {"output", OutputSettings::schema("Output settings patch")},
                      {"interfaces", {{"type", "array"}, {"items", InterfaceSettings::schema("Directed interface patch")}}}}}};
        return schema;
    }

    SimulationConfig SimulationConfig::Resolve(const Json &input, int size, const std::vector<ConfigOverride> &overrides)
    {
        DNDS_check_throw_info(size > 0, "invalid simulation size");
        Config::ValidatePartialJsonSchema(input, InputSchema());
        DNDS_check_throw_info(input.value("schemaVersion", 1) == 1, "unsupported Cauchy schema version");
        const auto supplied = [&](const char *section, const char *field)
        {
            const std::string pointer = std::string("/") + section + "/" + field;
            return (input.contains(section) && input.at(section).contains(field)) ||
                   std::any_of(overrides.begin(), overrides.end(), [&](const auto &override)
                               { return override.pointer == pointer; });
        };
        const auto activeDeprecatedAlias = [&](const char *section, const char *field)
        {
            const std::string pointer = std::string("/") + section + "/" + field;
            const auto active = [](const Json &value)
            { return value.is_number() && value.get<real>() >= 0; };
            if (input.contains(section) && input.at(section).contains(field) && active(input.at(section).at(field)))
                return true;
            return std::any_of(overrides.begin(), overrides.end(), [&](const auto &override)
                               { return override.pointer == pointer && active(override.value); });
        };
        DNDS_check_throw_info(!(supplied("time", "tEnd") && activeDeprecatedAlias("time", "finalTime")),
                              "specify only time.tEnd or deprecated time.finalTime");
        DNDS_check_throw_info(!(supplied("output", "checkpointInterval") && activeDeprecatedAlias("output", "interval")),
                              "specify only output.checkpointInterval or deprecated output.interval");
        const auto schema = DomainSettings::schema();
        const auto profiles = input.value("profiles", Json::object());
        DNDS_check_throw_info(profiles.is_object(), "Cauchy profiles must be named objects");
        for (const auto &[name, patch] : profiles.items())
            Config::ValidatePartialJsonSchema(patch, schema, "/profiles/" + Escape(name));
        const auto defaults = input.value("defaults", Json::object());
        Config::ValidatePartialJsonSchema(defaults, schema, "/defaults");
        std::vector<int> allRanks(size);
        std::iota(allRanks.begin(), allRanks.end(), 0);
        const auto domains = input.value("domains", Json::array({Json{{"name", "fluid"}, {"ranks", allRanks}}}));
        DNDS_check_throw_info(domains.is_array() && !domains.empty(), "Cauchy needs at least one domain");
        auto data = std::make_shared<Data>();
        data->root = Json{{"schemaVersion", 1}, {"domains", Json::array()}, {"time", TimeSettings{}}, {"output", OutputSettings{}}};
        Origin(data->root, "", "schema default", data->provenance);
        data->provenance.erase("/domains");
        for (index i = 0; i < index(domains.size()); ++i)
        {
            const auto &source = domains[i];
            const auto path = "/domains/" + std::to_string(i);
            Json settings = DomainSettings{};
            Origin(settings, path + "/settings", "schema default", data->provenance);
            const auto selected = source.value("profiles", std::vector<std::string>{});
            std::set<std::string> used;
            for (const auto &name : selected)
            {
                DNDS_check_throw_info(profiles.contains(name) && used.insert(name).second, "unknown/duplicate Cauchy profile " + name);
                Merge(settings, profiles[name], path + "/settings", "profile " + name, data->provenance);
            }
            Merge(settings, defaults, path + "/settings", "solver defaults", data->provenance);
            const auto patch = source.value("overrides", Json::object());
            Config::ValidatePartialJsonSchema(patch, schema, path + "/overrides");
            Merge(settings, patch, path + "/settings", "domain override", data->provenance);
            // Map entries are partial conditions too. Fill defaults without
            // overwriting the per-leaf origin of explicitly supplied values.
            for (auto &[zone, condition] : settings["boundaries"].items())
            {
                const Json original = condition;
                condition = BoundarySettings{};
                for (const auto &[key, value] : condition.items())
                    if (!original.contains(key))
                        Origin(value, path + "/settings/boundaries/" + Escape(zone) + "/" + key, "schema default", data->provenance);
                condition.update(original);
                // A map value is a partial object, including nested profile
                // sections. Resolve their defaults just like root sections.
                if (original.contains("profile"))
                {
                    Json profile = FlowProfileSettings{};
                    for (const auto &[key, value] : profile.items())
                        if (!original["profile"].contains(key))
                            Origin(value, path + "/settings/boundaries/" + Escape(zone) + "/profile/" + key, "schema default", data->provenance);
                    profile.update(original["profile"]);
                    condition["profile"] = std::move(profile);
                }
            }
            Json resolved{{"name", source.at("name")}, {"ranks", source.value("ranks", allRanks)}, {"settings", std::move(settings)}};
            Origin(resolved["name"], path + "/name", "domain descriptor", data->provenance);
            Origin(resolved["ranks"], path + "/ranks", "domain descriptor", data->provenance);
            data->root["domains"].push_back(std::move(resolved));
        }
        for (const auto *section : {"time", "output"})
            if (input.contains(section))
                Merge(data->root[section], input[section], std::string("/") + section, "solver override", data->provenance);
        if (input.contains("interfaces"))
        {
            DNDS_check_throw_info(input["interfaces"].is_array(), "interfaces must be an array of directed connections");
            data->root["interfaces"] = Json::array();
            for (const auto &entry : input["interfaces"])
            {
                Config::ValidatePartialJsonSchema(entry, InterfaceSettings::schema(), "/interfaces");
                Json resolved = InterfaceSettings{};
                resolved.update(entry);
                data->root["interfaces"].push_back(std::move(resolved));
            }
            Origin(data->root["interfaces"], "/interfaces", "solver override", data->provenance);
        }
        for (const auto &override : overrides)
        {
            const Json::json_pointer pointer(override.pointer);
            DNDS_check_throw_info(!override.pointer.empty() && data->root.contains(pointer), "CLI override must target an existing resolved key: " + override.pointer);
            Merge(data->root[pointer], override.value, override.pointer, "CLI " + override.pointer, data->provenance);
        }
        DNDS_check_throw_info(data->root["schemaVersion"] == 1, "unsupported resolved schema version");
        DNDS_check_throw_info(data->root["domains"].is_array() && !data->root["domains"].empty(), "resolved Cauchy configuration has no domains");
        std::set<std::string> names;
        for (index iDomain = 0; iDomain < index(data->root["domains"].size()); ++iDomain)
        {
            auto &resolved = data->root["domains"][iDomain];
            Allowed(resolved, {"name", "ranks", "settings"}, "/domains");
            Config::ValidatePartialJsonSchema(resolved.at("settings"), schema, "/domains/settings");
            // Coarse levels are dynamic map entries, so root defaults cannot
            // supply their omitted count/solver fields. Expand after CLI patches
            // as well; the solver patch itself stays partial for inheritance.
            for (auto &[name, level] : resolved["settings"]["implicit"]["multigrid"]["coarseLevelSettings"].items())
            {
                const Json original = level;
                level = MultigridLevelSettings{};
                for (const auto &[key, value] : level.items())
                    if (!original.contains(key))
                        Origin(value, "/domains/" + std::to_string(iDomain) + "/settings/implicit/multigrid/coarseLevelSettings/" + Escape(name) + "/" + key,
                               "schema default", data->provenance);
                level.update(original);
            }
            DomainConfig domain{{resolved.at("name").get<std::string>(), resolved.at("ranks").get<std::vector<int>>()}, resolved.at("settings").get<DomainSettings>()};
            const bool requestedCoreFirst = domain.settings.numerics.cfv.coreFirst;
            domain.settings = ResolveSweepOrdering(std::move(domain.settings));
            if (requestedCoreFirst != domain.settings.numerics.cfv.coreFirst)
            {
                resolved["settings"]["numerics"]["cfv"]["coreFirst"] = domain.settings.numerics.cfv.coreFirst;
                data->provenance["/domains/" + std::to_string(iDomain) + "/settings/numerics/cfv/coreFirst"] = "Cauchy SOR logical ordering";
            }
            DNDS_check_throw_info(names.insert(domain.placement.name).second, "duplicate Cauchy domain name");
            Validate(domain, size);
            ResolvedExecutionPlan::Resolve(domain);
            data->domains.push_back(std::move(domain));
        }
        Config::ValidatePartialJsonSchema(data->root["time"], TimeSettings::schema(), "/time");
        Config::ValidatePartialJsonSchema(data->root["output"], OutputSettings::schema(), "/output");
        data->time = data->root["time"].get<TimeSettings>();
        data->output = data->root["output"].get<OutputSettings>();
        if (data->time.finalTime >= 0)
        {
            DNDS_check_throw_info(data->time.tEnd == 0 || data->time.tEnd == data->time.finalTime,
                                  "time.tEnd and deprecated time.finalTime differ");
            data->time.tEnd = data->time.finalTime;
            data->time.finalTime = -1;
            data->root["time"]["tEnd"] = data->time.tEnd;
            data->root["time"]["finalTime"] = -1;
            data->provenance["/time/tEnd"] = "deprecated /time/finalTime alias";
        }
        if (data->output.interval >= 0)
        {
            DNDS_check_throw_info(data->output.checkpointInterval == OutputSettings{}.checkpointInterval ||
                                      data->output.checkpointInterval == data->output.interval,
                                  "output.checkpointInterval and deprecated output.interval differ");
            data->output.checkpointInterval = data->output.interval;
            data->output.interval = -1;
            data->root["output"]["checkpointInterval"] = data->output.checkpointInterval;
            data->root["output"]["interval"] = -1;
            data->provenance["/output/checkpointInterval"] = "deprecated /output/interval alias";
        }
        std::set<std::pair<std::string, int>> receivers;
        for (const auto &entry : data->root.value("interfaces", Json::array()))
        {
            Config::ValidatePartialJsonSchema(entry, InterfaceSettings::schema(), "/interfaces");
            auto connection = entry.get<InterfaceSettings>();
            DNDS_check_throw_info(names.count(connection.donor) && names.count(connection.receiver) && connection.donor != connection.receiver,
                                  "interface donor/receiver names must identify distinct configured domains");
            DNDS_check_throw_info(connection.routing == "coarseBoxes" || connection.routing == "allGatherBoundary", "unknown interface routing mode");
            DNDS_check_throw_info(connection.coarseBoxesPerRank > 0 && std::isfinite(connection.maxGap) && connection.maxGap >= 0, "invalid interface search bounds");
            DNDS_check_throw_info(receivers.emplace(connection.receiver, connection.receiverZone).second, "interface receiver zone has multiple incoming connections");
            const auto receiver = std::find_if(data->domains.begin(), data->domains.end(), [&](const auto &domain)
                                               { return domain.placement.name == connection.receiver; });
            const auto donor = std::find_if(data->domains.begin(), data->domains.end(), [&](const auto &domain)
                                            { return domain.placement.name == connection.donor; });
            DNDS_check_throw_info(donor->settings.equation.eos == receiver->settings.equation.eos &&
                                      donor->settings.equation.gamma == receiver->settings.equation.gamma && donor->settings.equation.gasConstant == receiver->settings.equation.gasConstant,
                                  "initial flow5 interfaces require the same gas/energy convention on both domains");
            const auto condition = receiver->settings.boundaries.find(std::to_string(connection.receiverZone));
            DNDS_check_throw_info(condition != receiver->settings.boundaries.end() && condition->second.kind == "interface", "receiver zone must have an interface boundary condition");
            data->interfaces.push_back(std::move(connection));
        }
        for (const auto &domain : data->domains)
            for (const auto &[zone, condition] : domain.settings.boundaries)
                DNDS_check_throw_info(condition.kind != "interface" || receivers.count({domain.placement.name, std::stoi(zone)}), "interface boundary has no incoming connection");
        const std::vector<std::string> timeMethods{"ssprk3", "bdf1", "bdf2", "trapz", "esdirk2", "esdirk3", "esdirk4", "ditr"};
        DNDS_check_throw_info(std::find(timeMethods.begin(), timeMethods.end(), data->time.method) != timeMethods.end() && std::isfinite(data->time.dt) && data->time.dt > 0 && data->time.steps >= 0 &&
                                  std::isfinite(data->time.tEnd) && data->time.tEnd >= 0 && data->time.finalTime == -1 && std::isfinite(data->time.cfl) && data->time.cfl >= 0 &&
                                  std::isfinite(data->time.wallSeconds) && data->time.wallSeconds >= 0 && data->time.maxRetries >= 0,
                              "Cauchy requires an available time method, positive dt and nonnegative steps");
        DNDS_check_throw_info(std::isfinite(data->time.steadyTolerance) && data->time.steadyTolerance >= 0 && data->time.steadyMinSteps >= 0, "invalid steady stopping controls");
        data->time.ditr.Validate();
        data->time.positivity.Validate();
        data->time.ValidateSourceIntegration();
        if (data->time.sourceSplitting == "strang")
            DNDS_check_throw_info(std::any_of(data->domains.begin(), data->domains.end(), [](const auto &domain)
                                              { return domain.settings.equation.source == "chemical"; }),
                                  "Strang source splitting requires a configured chemical source");
        if (data->time.steady)
        {
            if (data->time.positivity.enabled)
            {
                data->time.positivity.enabled = false;
                data->root["time"]["positivity"]["enabled"] = false;
                data->provenance["/time/positivity/enabled"] = "BDF1 steady limit";
            }
            data->time.method = "bdf1";
            data->time.steps = std::min(index(1), data->time.steps);
            data->time.tEnd = data->time.cfl = 0;
            data->time.maxRetries = 0;
            data->time.steadyMinSteps = std::min(index(1), data->time.steadyMinSteps);
            const Json effective = data->time;
            for (const auto *name : {"method", "steps", "tEnd", "cfl", "maxRetries", "steadyMinSteps"})
                if (data->root["time"][name] != effective[name])
                {
                    data->root["time"][name] = effective[name];
                    data->provenance[std::string("/time/") + name] = "BDF1 steady limit";
                }
        }
        for (const auto &domain : data->domains)
        {
            DNDS_check_throw_info(data->time.method != "ssprk3" || !domain.settings.implicit.multigrid.levels, "polynomial FAS requires an implicit time method");
            if (data->time.method == "ditr")
                data->time.ditr.ResolveControls(domain.settings.implicit);
        }
        const std::vector<std::string> diagnosticLevels{"trace", "debug", "info", "warn", "error", "critical", "off"};
        DNDS_check_throw_info(data->output.interval == -1 && data->output.checkpointInterval >= 0 && data->output.visualizationInterval >= -1 &&
                                  std::isfinite(data->output.checkpointTimeInterval) && data->output.checkpointTimeInterval >= 0 &&
                                  std::isfinite(data->output.visualizationTimeInterval) && data->output.visualizationTimeInterval >= -1 &&
                                  std::isfinite(data->output.endpointSnapRelativeTolerance) && data->output.endpointSnapRelativeTolerance >= 0 &&
                                  data->output.consoleInterval >= 0 && data->output.historyInterval > 0 && data->output.historySegmentRows > 0 &&
                                  data->output.historyPrecision >= 1 && data->output.historyPrecision <= std::numeric_limits<real>::max_digits10 &&
                                  data->output.implicit.consoleInterval >= 0 && data->output.implicit.historyInterval > 0 &&
                                  data->output.implicit.checkpointInterval >= 0 && data->output.implicit.visualizationInterval >= -1 &&
                                  (data->output.consoleColor == "auto" || data->output.consoleColor == "always" || data->output.consoleColor == "never") &&
                                  std::find(diagnosticLevels.begin(), diagnosticLevels.end(), data->output.diagnostics.level) != diagnosticLevels.end() &&
                                  data->output.diagnostics.maxBytes > 0 && data->output.diagnostics.rotatedFiles > 0 && !data->output.prefix.empty(),
                              "invalid Cauchy output controls");
        DNDS_check_throw_info(data->output.hdfDeflateLevel >= 0 && data->output.hdfDeflateLevel <= 9, "Cauchy output hdfDeflateLevel must be in [0,9]");
        const auto internalVisualization = data->output.implicit.visualizationInterval < 0
                                               ? data->output.implicit.checkpointInterval
                                               : data->output.implicit.visualizationInterval;
        DNDS_check_throw_info(data->output.implicit.checkpointInterval == 0 || data->domains.size() == 1,
                              "exact internal checkpoints currently require one domain");
        DNDS_check_throw_info((data->output.implicit.checkpointInterval == 0 && internalVisualization == 0) ||
                                  data->time.steady || data->time.method != "ssprk3",
                              "internal-iteration output requires an implicit method");
        return SimulationConfig(std::move(data));
    }
    std::vector<DomainRankGroup> SimulationConfig::RankGroups() const
    {
        std::vector<DomainRankGroup> result;
        for (const auto &domain : _data->domains)
            result.push_back(domain.placement);
        return result;
    }
    SimulationConfig SimulationConfig::ResolveCollective(const Json &input, MPI_Comm comm, const std::vector<ConfigOverride> &overrides)
    {
        DNDS_check_throw_info(comm != MPI_COMM_NULL, "Cauchy config needs a simulation communicator");
        int size = 0, rank = 0, valid = 1;
        MPI_Comm_size(comm, &size);
        MPI_Comm_rank(comm, &rank);
        std::optional<SimulationConfig> result;
        std::string error, serialized;
        try
        {
            result = Resolve(input, size, overrides);
            serialized = Json{{"root", result->Root()}, {"provenance", result->Provenance()}}.dump();
            DNDS_check_throw_info(serialized.size() <= 16 * 1024 * 1024, "Cauchy resolved configuration exceeds 16 MiB");
        }
        catch (const std::exception &exception)
        {
            MPI::AbortIfSolverFailure("Cauchy configuration rejected");
            error = exception.what();
            valid = 0;
        }
        valid = MPI::AllreduceValue<MPI::Reduction::Min>(valid, comm);
        DNDS_check_throw_info(valid, "Cauchy configuration rejected on a rank: " + error);
        int length = rank == 0 ? int(serialized.size()) : 0;
        MPI_Bcast(&length, 1, MPI_INT, 0, comm);
        std::string reference = rank == 0 ? serialized : std::string(size_t(length), '\0');
        MPI_Bcast(reference.data(), length, MPI_CHAR, 0, comm);
        valid = reference == serialized;
        valid = MPI::AllreduceValue<MPI::Reduction::Min>(valid, comm);
        MPI::CheckPreflight(valid, "Cauchy resolved configuration differs between simulation ranks");
        return std::move(*result);
    }
}
