#include "Cauchy/Config/SimulationConfig.hpp"
#include "Cauchy/Physics/StateInput.hpp"
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
        bool MatchesType(const Json &value, const std::string &type)
        {
            return type.empty() || (type == "null" && value.is_null()) || (type == "object" && value.is_object()) ||
                   (type == "array" && value.is_array()) || (type == "string" && value.is_string()) ||
                   (type == "boolean" && value.is_boolean()) || (type == "integer" && value.is_number_integer()) ||
                   (type == "number" && value.is_number() && std::isfinite(value.get<real>()));
        }
        void Keys(const Json &value, const Json &schema, const std::string &path)
        {
            if (schema.contains("oneOf"))
            {
                // Input unions distinguish null/array/object. Select by JSON
                // type without raising diagnostic exceptions on valid input.
                const Json *selected = nullptr;
                for (const auto &variant : schema["oneOf"])
                    if (MatchesType(value, variant.value("type", std::string{})))
                    {
                        DNDS_check_throw_info(!selected, "ambiguous Cauchy configuration variant at " + path);
                        selected = &variant;
                    }
                DNDS_check_throw_info(selected, "Cauchy configuration does not match an input variant at " + path);
                Keys(value, *selected, path);
                return;
            }
            const auto type = schema.value("type", std::string{});
            DNDS_check_throw_info(MatchesType(value, type), "Cauchy configuration type mismatch at " + path);
            if (schema.contains("enum"))
                DNDS_check_throw_info(std::find(schema["enum"].begin(), schema["enum"].end(), value) != schema["enum"].end(), "unknown Cauchy enum value at " + path);
            if (schema.contains("required"))
                for (const auto &key : schema["required"])
                    DNDS_check_throw_info(value.contains(key.get<std::string>()), "missing Cauchy configuration field at " + path + "/" + key.get<std::string>());
            if (value.is_object())
                for (const auto &[key, child] : value.items())
                {
                    const auto childPath = path + "/" + Escape(key);
                    if (schema.contains("properties") && schema["properties"].contains(key))
                        Keys(child, schema["properties"][key], childPath);
                    else if (schema.contains("additionalProperties") && schema["additionalProperties"].is_object())
                        Keys(child, schema["additionalProperties"], childPath);
                    else
                        DNDS_check_throw_info(false, "unknown Cauchy configuration key " + childPath);
                }
            if (value.is_array())
            {
                DNDS_check_throw_info(value.size() >= schema.value("minItems", size_t(0)) && value.size() <= schema.value("maxItems", SIZE_MAX),
                                      "Cauchy array extent mismatch at " + path);
                if (schema.contains("items"))
                    for (index i = 0; i < index(value.size()); ++i)
                    {
                        const auto &items = schema["items"];
                        if (items.is_object())
                            Keys(value[i], items, path + "/" + std::to_string(i));
                        else if (items.is_array() && i < index(items.size()))
                            Keys(value[i], items[i], path + "/" + std::to_string(i));
                    }
            }
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
                                  (positivityReconstruction == "Euler" || positivityReconstruction == "Segment") && positivityBetaPower >= 0 && positivityBetaPower < INT_MAX,
                              "invalid reconstruction positivity controls");
        DNDS_check_throw_info(!positivity || positivityReconstruction != "Segment" || (positivityCompressToMean && positivityBetaPower == 1),
                              "Segment compatibility does not implement Euler order-reference or beta-power controls");
    }
    void DITRTimeSettings::Validate() const
    {
        DNDS_check_throw_info((variant == "U2R2" || variant == "U2R1") && std::isfinite(fraction) && fraction > 0 && fraction < real(2) / 3 &&
                                  std::isfinite(theta) && theta > 0 && iterations > 0 && minIterations >= 0 && minIterations <= iterations &&
                                  std::isfinite(relativeTolerance) && relativeTolerance >= 0 && std::isfinite(absoluteTolerance) && absoluteTolerance >= 0,
                              "invalid/unavailable DITR positive-stage-scale controls");
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
                                  (jacobianAction == "assembled" || jacobianAction == "semiMatrixFree") &&
                                  iterations > 0 && minIterations >= 0 && minIterations <= iterations && consoleInterval >= 0 &&
                                  nonnegative(relativeTolerance) && nonnegative(absoluteTolerance) &&
                                  std::isfinite(pseudoCFL) && pseudoCFL > 0 && std::isfinite(maxPseudoCFL) && maxPseudoCFL >= pseudoCFL &&
                                  std::isfinite(pseudoGrowth) && pseudoGrowth >= 1 && std::isfinite(relaxation) && relaxation > 0 && relaxation <= 1,
                              "invalid implicit nonlinear/diagonal/pseudo-time controls");
        DNDS_check_throw_info((linear.method == "sgs" || linear.method == "gmres") && (linear.preconditioner == "sgs" || linear.preconditioner == "ilu") &&
                                  linear.iterations > 0 && linear.subspace > 0 && linear.subspace < INT_MAX &&
                                  nonnegative(linear.relativeTolerance) && nonnegative(linear.absoluteTolerance) && sgs.sweeps > 0 && ilu.fill >= -1 && ilu.fill <= 5 &&
                                  !(linear.method == "gmres" && linear.preconditioner == "ilu" && sgs.exchangeBetweenSweeps),
                              "invalid implicit linear/SGS/ILU controls");
        DNDS_check_throw_info((positivity.rhsMode == "none" || positivity.rhsMode == "rhs" || positivity.rhsMode == "dTau") &&
                                  (positivity.rhsReduction == "cell" || positivity.rhsReduction == "minimumLimited" || positivity.rhsReduction == "minimumAll") &&
                                  nonnegative(positivity.rhsScale) && std::isfinite(positivity.rhsRelax) && positivity.rhsRelax > 0 && positivity.rhsRelax <= 1 &&
                                  std::isfinite(positivity.incrementRelax) && positivity.incrementRelax > 0 && positivity.incrementRelax <= 1 &&
                                  positivity.compressionPower > 0 && positivity.compressionPower <= INT_MAX / 2,
                              "invalid implicit Euler-PP controls");
        DNDS_check_throw_info(multigrid.levels >= 0 && multigrid.levels <= 2 && multigrid.cycles > 0 &&
                                  multigrid.preSmooth >= 0 && multigrid.postSmooth >= 0 && multigrid.coarseSmooth > 0 &&
                                  (!multigrid.levels || (multigrid.preSmooth <= iterations && multigrid.postSmooth <= iterations && multigrid.coarseSmooth <= iterations)) &&
                                  std::isfinite(multigrid.damping) && multigrid.damping > 0 && multigrid.damping <= 1,
                              "invalid polynomial FAS controls");
    }
    DomainSettings ResolveSweepOrdering(DomainSettings settings)
    {
        if (settings.numerics.cfv.SORInstead)
            settings.numerics.cfv.coreFirst = true;
        return settings;
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
        require(s.equation.system == "compressible", "equation " + s.equation.system);
        const bool mixture = s.equation.eos == "idealGasMixture";
        const bool rans = s.equation.rans.model != "none", extended = mixture || rans;
        require(rans || s.initialRANS.empty(), "initialRANS without a RANS model");
        if (rans)
        {
            require((s.equation.rans.model == "SA" || s.equation.rans.model == "KOmegaWilcox" ||
                     s.equation.rans.model == "KOmegaSST" || s.equation.rans.model == "RealizableKE") &&
                        s.equation.rans.desScale > 100,
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
            require(s.equation.viscosity == 0, "mixture transport uses its material, not constant viscosity");
        }
        if (extended)
        {
            require(s.initialProfile.kind == "constant", "nonconstant extended equation initialization");
            require(Riemann::RiemannFlux::Resolve(s.numerics.flux).Kind() == Riemann::Solver::Rusanov && !s.numerics.faceFallback,
                    "extended equation flux preparation/face fallback");
            require(s.numerics.limiter == "none" || (s.numerics.limiter == "WBAP" && !s.numerics.characteristicLimiter), "extended equation limiter");
            require(s.equation.transport != "NewtonianFourier" || s.numerics.viscousGradient == "polynomial", "extended equation viscous gradient");
            require(!s.implicit.multigrid.levels && !s.implicit.roeDissipation, "extended equation p-FAS/Roe Jacobian");
        }
        require(s.initialField.Empty() || s.initialProfile.kind == "constant", "spatial overrides require a constant background profile");
        require(Riemann::RiemannFlux::Available(s.numerics.flux), "numerical flux " + s.numerics.flux);
        require(s.numerics.reconstruction == "constant" || s.numerics.reconstruction == "gradient" || s.numerics.reconstruction == "variational", "reconstruction " + s.numerics.reconstruction);
        require(s.numerics.gradient == "leastSquares" || s.numerics.gradient == "greenGauss" || s.numerics.gradient == "ggmp", "gradient " + s.numerics.gradient);
        require(s.numerics.viscousGradient == "primitive" || s.numerics.viscousGradient == "polynomial", "viscous gradient " + s.numerics.viscousGradient);
        require(s.numerics.limiter == "none" || s.numerics.limiter == "WBAP" || s.numerics.limiter == "LVR", "limiter " + s.numerics.limiter);
        require(s.numerics.limiter != "WBAP" || s.numerics.reconstruction != "constant", "WBAP needs polynomial reconstruction");
        require(s.numerics.limiter != "LVR" || (s.numerics.reconstruction == "variational" && s.numerics.cfv.maxOrder >= 2), "LVR needs degree2/3 variational reconstruction");
        require(!s.implicit.multigrid.levels || (s.numerics.reconstruction == "variational" && s.numerics.cfv.maxOrder >= 2), "polynomial FAS requires high-order variational reconstruction");
        for (const auto &[zone, condition] : s.boundaries)
        {
            require(condition.kind == "extrapolate" || condition.kind == "farfield" || condition.kind == "slipWall" || condition.kind == "noSlipWall" || condition.kind == "analytic" || condition.kind == "doubleMachBottom" || condition.kind == "interface", "BC zone " + zone + ": " + condition.kind);
            require(condition.kind != "doubleMachBottom" || condition.profile.kind == "doubleMachReflection", "DMR bottom requires its analytic shock profile");
            require(condition.thermal == "adiabatic" || condition.thermal == "isothermal", "thermal BC " + condition.thermal);
            require(condition.wallVelocityFrame == "mesh" || condition.wallVelocityFrame == "ground", "wall velocity frame " + condition.wallVelocityFrame);
            require(!extended || (condition.kind != "interface" && condition.kind != "doubleMachBottom" && (condition.kind != "analytic" || condition.profile.kind == "constant")),
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
    SimulationConfig SimulationConfig::Resolve(const Json &input, int size, const std::vector<ConfigOverride> &overrides)
    {
        DNDS_check_throw_info(size > 0, "invalid simulation size");
        Allowed(input, {"schemaVersion", "profiles", "defaults", "domains", "time", "output", "interfaces"}, "");
        DNDS_check_throw_info(input.value("schemaVersion", 1) == 1, "unsupported Cauchy schema version");
        const auto schema = DomainSettings::schema();
        const auto profiles = input.value("profiles", Json::object());
        DNDS_check_throw_info(profiles.is_object(), "Cauchy profiles must be named objects");
        for (const auto &[name, patch] : profiles.items())
            Keys(patch, schema, "/profiles/" + Escape(name));
        const auto defaults = input.value("defaults", Json::object());
        Keys(defaults, schema, "/defaults");
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
            Allowed(source, {"name", "ranks", "profiles", "overrides"}, path);
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
            Keys(patch, schema, path + "/overrides");
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
                Keys(entry, InterfaceSettings::schema(), "/interfaces");
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
            Keys(resolved.at("settings"), schema, "/domains/settings");
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
        Keys(data->root["time"], TimeSettings::schema(), "/time");
        Keys(data->root["output"], OutputSettings::schema(), "/output");
        data->time = data->root["time"].get<TimeSettings>();
        data->output = data->root["output"].get<OutputSettings>();
        std::set<std::pair<std::string, int>> receivers;
        for (const auto &entry : data->root.value("interfaces", Json::array()))
        {
            Keys(entry, InterfaceSettings::schema(), "/interfaces");
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
        DNDS_check_throw_info(data->interfaces.empty() || data->time.steady || data->time.method != "ditr", "multi-domain DITR requires coupled midpoint/end coordination");
        const std::vector<std::string> timeMethods{"ssprk3", "bdf1", "bdf2", "trapz", "esdirk2", "esdirk3", "esdirk4", "ditr"};
        DNDS_check_throw_info(std::find(timeMethods.begin(), timeMethods.end(), data->time.method) != timeMethods.end() && std::isfinite(data->time.dt) && data->time.dt > 0 && data->time.steps >= 0 &&
                                  std::isfinite(data->time.finalTime) && data->time.finalTime >= 0 && std::isfinite(data->time.cfl) && data->time.cfl >= 0 &&
                                  std::isfinite(data->time.wallSeconds) && data->time.wallSeconds >= 0 && data->time.maxRetries >= 0,
                              "Cauchy requires an available time method, positive dt and nonnegative steps");
        DNDS_check_throw_info(std::isfinite(data->time.steadyTolerance) && data->time.steadyTolerance >= 0 && data->time.steadyMinSteps >= 0, "invalid steady stopping controls");
        data->time.ditr.Validate();
        data->time.positivity.Validate();
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
            data->time.finalTime = data->time.cfl = 0;
            data->time.maxRetries = 0;
            data->time.steadyMinSteps = std::min(index(1), data->time.steadyMinSteps);
            const Json effective = data->time;
            for (const auto *name : {"method", "steps", "finalTime", "cfl", "maxRetries", "steadyMinSteps"})
                if (data->root["time"][name] != effective[name])
                {
                    data->root["time"][name] = effective[name];
                    data->provenance[std::string("/time/") + name] = "BDF1 steady limit";
                }
        }
        for (const auto &domain : data->domains)
        {
            DNDS_check_throw_info(data->time.method != "ssprk3" || !domain.settings.implicit.multigrid.levels, "polynomial FAS requires an implicit time method");
            if (domain.settings.equation.eos == "idealGasMixture" || domain.settings.equation.rans.model != "none")
                DNDS_check_throw_info(!data->time.positivity.enabled && !(data->time.method == "ssprk3" && domain.settings.numerics.positivity),
                                      "extended equation physical-dt and explicit flux-blend PP adapters are not connected yet");
        }
        DNDS_check_throw_info(data->output.interval >= 0 && data->output.historyInterval > 0 && data->output.historySegmentRows > 0 && !data->output.prefix.empty(), "invalid Cauchy output controls");
        DNDS_check_throw_info(data->output.hdfDeflateLevel >= 0 && data->output.hdfDeflateLevel <= 9, "Cauchy output hdfDeflateLevel must be in [0,9]");
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
            error = exception.what();
            valid = 0;
        }
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, comm);
        DNDS_check_throw_info(valid, "Cauchy configuration rejected on a rank: " + error);
        int length = rank == 0 ? int(serialized.size()) : 0;
        MPI_Bcast(&length, 1, MPI_INT, 0, comm);
        std::string reference = rank == 0 ? serialized : std::string(size_t(length), '\0');
        MPI_Bcast(reference.data(), length, MPI_CHAR, 0, comm);
        valid = reference == serialized;
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, comm);
        DNDS_check_throw_info(valid, "Cauchy resolved configuration differs between simulation ranks");
        return std::move(*result);
    }
}
