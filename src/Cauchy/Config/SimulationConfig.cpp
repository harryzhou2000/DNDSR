#include "Cauchy/Config/SimulationConfig.hpp"
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
        void Keys(const Json &value, const Json &schema, const std::string &path)
        {
            const auto type = schema.value("type", std::string{});
            bool valid = type.empty() || (type == "object" && value.is_object()) ||
                         (type == "array" && value.is_array()) || (type == "string" && value.is_string()) ||
                         (type == "boolean" && value.is_boolean()) || (type == "integer" && value.is_number_integer()) ||
                         (type == "number" && value.is_number() && std::isfinite(value.get<real>()));
            DNDS_check_throw_info(valid, "Cauchy configuration type mismatch at " + path);
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
            DNDS_check_throw_info(!domain.placement.name.empty() && !domain.placement.ranks.empty(), "Cauchy domain needs a name and ranks");
            std::set<int> ranks;
            for (int rank : domain.placement.ranks)
                DNDS_check_throw_info(rank >= 0 && rank < size && ranks.insert(rank).second, "invalid/duplicate Cauchy domain rank");
            const auto &equation = settings.equation;
            DNDS_check_throw_info(std::isfinite(equation.gamma) && equation.gamma > 1 && std::isfinite(equation.gasConstant) && equation.gasConstant > 0 &&
                                      std::isfinite(equation.viscosity) && equation.viscosity >= 0 && std::isfinite(equation.prandtl) && equation.prandtl > 0,
                                  "invalid Cauchy constitutive constants");
            DNDS_check_throw_info(equation.transport != "inviscid" || equation.viscosity == 0, "inviscid configuration has unused nonzero viscosity");
            DNDS_check_throw_info(settings.mesh.dimension == 2 || settings.mesh.dimension == 3, "Cauchy geometric dimension must be 2 or 3");
            DNDS_check_throw_info(settings.execution.hostThreads >= 0 && settings.execution.hostThreads <= 1024, "invalid CPU worker count");
            const auto &numerics = settings.numerics;
            DNDS_check_throw_info(std::isfinite(numerics.temperatureFloor) && numerics.temperatureFloor > 0, "positive finite temperatureFloor required");
            DNDS_check_throw_info(std::isfinite(numerics.positivityBetaCutoff) && numerics.positivityBetaCutoff >= 0 && numerics.positivityBetaCutoff < 1, "invalid positivity beta cutoff");
            DNDS_check_throw_info(std::isfinite(numerics.densityFloor) && numerics.densityFloor > 0 && std::isfinite(numerics.pressureFloor) && numerics.pressureFloor > 0 &&
                                      numerics.reconstructionIterations > 0 && std::isfinite(numerics.reconstructionTolerance) && numerics.reconstructionTolerance >= 0,
                                  "invalid Cauchy numerical controls");
            numerics.cfv.Validate();
            const auto &riemann = numerics.riemann;
            DNDS_check_throw_info((numerics.flux != "Roe_M3" && numerics.flux != "Roe_M5") || riemann.fixScale > 0,
                                  "low-dissipation Roe flux requires a positive fixScale divisor");
            DNDS_check_throw_info(std::isfinite(riemann.fixScale) && riemann.fixScale >= 0 && std::isfinite(riemann.incrementScale) && riemann.incrementScale >= 0 &&
                                      std::isfinite(riemann.dLambda) && riemann.dLambda >= 0 && std::isfinite(riemann.referenceMach) && riemann.referenceMach > 0 &&
                                      std::isfinite(riemann.pressureStencilMin) && riemann.pressureStencilMin > 0,
                                  "invalid Cauchy Riemann controls");
            PrimitiveValid(settings.initialPrimitive);
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
                validateProfile(condition.profile);
                if (condition.kind == "noSlipWall" && condition.thermal == "isothermal")
                    DNDS_check_throw_info(condition.temperature >= numerics.temperatureFloor, "wall temperature lies below the configured temperature floor");
                if (condition.kind == "farfield" && equation.eos == "perfectGas")
                    DNDS_check_throw_info(condition.primitive[4] / (condition.primitive[0] * equation.gasConstant) >= numerics.temperatureFloor,
                                          "farfield temperature lies below the configured temperature floor");
                DNDS_check_throw_info(std::isfinite(condition.temperature) && condition.temperature > 0 &&
                                          std::all_of(condition.velocity.begin(), condition.velocity.end(), [](real v)
                                                      { return std::isfinite(v); }),
                                      "invalid boundary state");
                DNDS_check_throw_info(condition.kind == "noSlipWall" || (condition.thermal == "adiabatic" && condition.velocity == std::array<real, 3>{0, 0, 0}),
                                      "wall thermal/velocity parameters supplied to a non-wall condition");
            }
        }
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
        require(s.equation.eos == "perfectGas", "EOS " + s.equation.eos);
        require(s.equation.transport == "inviscid" || s.equation.transport == "NewtonianFourier", "constitutive law " + s.equation.transport);
        require(s.equation.source == "none", "volume source " + s.equation.source);
        require(Riemann::RiemannFlux::Available(s.numerics.flux), "numerical flux " + s.numerics.flux);
        require(s.numerics.reconstruction == "constant" || s.numerics.reconstruction == "gradient" || s.numerics.reconstruction == "variational", "reconstruction " + s.numerics.reconstruction);
        require(s.numerics.gradient == "leastSquares" || s.numerics.gradient == "greenGauss" || s.numerics.gradient == "ggmp", "gradient " + s.numerics.gradient);
        require(s.numerics.limiter == "none" || s.numerics.limiter == "WBAP", "limiter " + s.numerics.limiter);
        require(s.numerics.limiter != "WBAP" || s.numerics.reconstruction != "constant", "WBAP needs polynomial reconstruction");
        require(!s.numerics.cfv.SORInstead, "Cauchy frozen-iterate reconstruction currently requires Jacobi");
        require(s.numerics.cfv.cacheDiffBase, "Cauchy residual requires cached face basis derivatives");
        for (const auto &[zone, condition] : s.boundaries)
        {
            require(condition.kind == "extrapolate" || condition.kind == "farfield" || condition.kind == "slipWall" || condition.kind == "noSlipWall" || condition.kind == "analytic" || condition.kind == "doubleMachBottom", "BC zone " + zone + ": " + condition.kind);
            require(condition.kind != "doubleMachBottom" || condition.profile.kind == "doubleMachReflection", "DMR bottom requires its analytic shock profile");
            require(condition.thermal == "adiabatic" || condition.thermal == "isothermal", "thermal BC " + condition.thermal);
            require(condition.kind != "noSlipWall" || s.equation.transport == "NewtonianFourier", "no-slip wall needs viscous transport");
        }
        if (!missing.empty())
        {
            std::string message = "Cauchy domain " + domain.placement.name + " has missing capabilities:";
            for (const auto &item : missing)
                message += "\n  " + item;
            DNDS_check_throw_info(false, message);
        }
        return {{"cpu/perfectGas/flow5", "cpu/" + s.numerics.flux, "cpu/" + s.equation.transport,
                 "cfv/" + s.numerics.reconstruction, "cfv/" + s.numerics.limiter, "runtime/mpi-direct"}};
    }
    SimulationConfig SimulationConfig::Resolve(const Json &input, int size, const std::vector<ConfigOverride> &overrides)
    {
        DNDS_check_throw_info(size > 0, "invalid simulation size");
        Allowed(input, {"schemaVersion", "profiles", "defaults", "domains", "time", "output"}, "");
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
        for (const auto &override : overrides)
        {
            const Json::json_pointer pointer(override.pointer);
            DNDS_check_throw_info(!override.pointer.empty() && data->root.contains(pointer), "CLI override must target an existing resolved key: " + override.pointer);
            Merge(data->root[pointer], override.value, override.pointer, "CLI " + override.pointer, data->provenance);
        }
        DNDS_check_throw_info(data->root["schemaVersion"] == 1, "unsupported resolved schema version");
        DNDS_check_throw_info(data->root["domains"].is_array() && !data->root["domains"].empty(), "resolved Cauchy configuration has no domains");
        std::set<std::string> names;
        for (const auto &resolved : data->root["domains"])
        {
            Allowed(resolved, {"name", "ranks", "settings"}, "/domains");
            Keys(resolved.at("settings"), schema, "/domains/settings");
            DomainConfig domain{{resolved.at("name").get<std::string>(), resolved.at("ranks").get<std::vector<int>>()}, resolved.at("settings").get<DomainSettings>()};
            DNDS_check_throw_info(names.insert(domain.placement.name).second, "duplicate Cauchy domain name");
            Validate(domain, size);
            ResolvedExecutionPlan::Resolve(domain);
            data->domains.push_back(std::move(domain));
        }
        Keys(data->root["time"], TimeSettings::schema(), "/time");
        Keys(data->root["output"], OutputSettings::schema(), "/output");
        data->time = data->root["time"].get<TimeSettings>();
        data->output = data->root["output"].get<OutputSettings>();
        DNDS_check_throw_info(data->time.method == "ssprk3" && std::isfinite(data->time.dt) && data->time.dt > 0 && data->time.steps >= 0 &&
                                  std::isfinite(data->time.finalTime) && data->time.finalTime >= 0 && std::isfinite(data->time.cfl) && data->time.cfl >= 0 &&
                                  std::isfinite(data->time.wallSeconds) && data->time.wallSeconds >= 0 && data->time.maxRetries >= 0,
                              "Cauchy requires SSPRK3, positive dt and nonnegative steps");
        DNDS_check_throw_info(std::isfinite(data->time.steadyTolerance) && data->time.steadyTolerance >= 0 && data->time.steadyMinSteps >= 0, "invalid steady stopping controls");
        DNDS_check_throw_info(data->output.interval >= 0 && data->output.historyInterval > 0 && !data->output.prefix.empty(), "invalid Cauchy output controls");
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
