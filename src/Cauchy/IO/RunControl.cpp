#include "Cauchy/IO/RunControl.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace DNDS::Cauchy
{
    namespace
    {
        struct FieldRule
        {
            std::string pointer;
            LiveConfigCategory category;
            std::string type; // "integer" or "number"
            bool hasMin = false;
            bool exclusiveMin = false;
            double minValue = 0;
            bool hasMax = false;
            double maxValue = 0;
            std::string defaultApplyAt;
            std::string description;
        };

        std::vector<FieldRule> BuildRules(index domainCount)
        {
            std::vector<FieldRule> rules{
                // Output-only controls (non-intrusive to numerics/solution path)
                {"/output/consoleInterval", LiveConfigCategory::Output, "integer", true, false, 0, false, 0, "step",
                 "Accepted-step console output interval (0 disables)"},
                {"/output/historyInterval", LiveConfigCategory::Output, "integer", true, false, 1, false, 0, "step",
                 "Accepted-step CSV history interval (>= 1)"},
                {"/output/checkpointInterval", LiveConfigCategory::Output, "integer", true, false, 0, false, 0, "step",
                 "Accepted-step periodic checkpoint interval (0 disables)"},
                {"/output/visualizationInterval", LiveConfigCategory::Output, "integer", true, false, -1, false, 0, "step",
                 "Accepted-step periodic visualization interval (-1 inherits checkpointInterval, 0 disables)"},
                {"/output/checkpointTimeInterval", LiveConfigCategory::Output, "number", true, false, 0, false, 0, "step",
                 "Physical-time checkpoint cadence (0 disables)"},
                {"/output/visualizationTimeInterval", LiveConfigCategory::Output, "number", true, false, -1, false, 0, "step",
                 "Physical-time visualization cadence (-1 inherits checkpointTimeInterval, 0 disables)"},
                {"/output/implicit/consoleInterval", LiveConfigCategory::Output, "integer", true, false, 0, false, 0, "iteration",
                 "Nonlinear iteration console cadence (0 suppresses ordinary lines)"},
                {"/output/implicit/historyInterval", LiveConfigCategory::Output, "integer", true, false, 1, false, 0, "iteration",
                 "Nonlinear iteration CSV & per-boundary flux cadence (>= 1)"},
                {"/output/implicit/checkpointInterval", LiveConfigCategory::Output, "integer", true, false, 0, false, 0, "iteration",
                 "Restartable nonlinear-iteration checkpoint cadence (0 disables; single-domain implicit only)"},
                {"/output/implicit/visualizationInterval", LiveConfigCategory::Output, "integer", true, false, -1, false, 0, "iteration",
                 "Nonlinear-iteration visualization cadence (-1 inherits checkpointInterval, 0 disables)"},

                // Simulation steering controls (affects numerical trajectory/stopping)
                {"/time/steps", LiveConfigCategory::Steering, "integer", true, false, 0, false, 0, "step",
                 "Accepted step target"},
                {"/time/tEnd", LiveConfigCategory::Steering, "number", true, false, 0, false, 0, "step",
                 "Physical-time target (0 uses step target)"},
                {"/time/wallSeconds", LiveConfigCategory::Steering, "number", true, false, 0, false, 0, "step",
                 "Graceful wall-clock budget in seconds (0 disables)"},
                {"/time/cfl", LiveConfigCategory::Steering, "number", true, false, 0, false, 0, "step",
                 "Physical CFL number (0 uses fixed dt)"},
                {"/time/dt", LiveConfigCategory::Steering, "number", true, true, 0, false, 0, "step",
                 "Physical timestep / maximum dt cap (> 0)"},
                {"/time/steadyTolerance", LiveConfigCategory::Steering, "number", true, false, 0, false, 0, "step",
                 "Relative steady residual convergence target (0 disables)"}};

            for (index d = 0; d < domainCount; ++d)
            {
                const std::string base = "/domains/" + std::to_string(d) + "/settings/implicit/";
                rules.push_back({base + "iterations", LiveConfigCategory::Steering, "integer", true, false, 1, false, 0, "iteration",
                                 "Maximum nonlinear corrections per physical stage (>= 1)"});
                rules.push_back({base + "minIterations", LiveConfigCategory::Steering, "integer", true, false, 0, false, 0, "iteration",
                                 "Minimum nonlinear corrections before accepting a stage (>= 0)"});
                rules.push_back({base + "pseudoCFL", LiveConfigCategory::Steering, "number", true, true, 0, false, 0, "iteration",
                                 "Current/initial pseudo-time CFL (> 0)"});
                rules.push_back({base + "maxPseudoCFL", LiveConfigCategory::Steering, "number", true, true, 0, false, 0, "iteration",
                                 "Maximum pseudo-time CFL cap (> 0)"});
                rules.push_back({base + "pseudoGrowth", LiveConfigCategory::Steering, "number", true, false, 1, false, 0, "iteration",
                                 "Pseudo-time CFL multiplicative growth factor (>= 1)"});
                rules.push_back({base + "pseudoGrowthStart", LiveConfigCategory::Steering, "integer", true, false, 0, false, 0, "iteration",
                                 "Nonlinear iteration index at which pseudo-CFL growth starts (>= 0)"});
                rules.push_back({base + "relaxation", LiveConfigCategory::Steering, "number", true, true, 0, true, 1, "iteration",
                                 "Flow correction relaxation factor in (0, 1]"});
                rules.push_back({base + "ransRelaxation", LiveConfigCategory::Steering, "number", true, true, 0, true, 1, "iteration",
                                 "RANS correction relaxation factor in (0, 1]"});
                rules.push_back({base + "relativeTolerance", LiveConfigCategory::Steering, "number", true, false, 0, false, 0, "iteration",
                                 "Relative nonlinear residual convergence tolerance (>= 0)"});
                rules.push_back({base + "absoluteTolerance", LiveConfigCategory::Steering, "number", true, false, 0, false, 0, "iteration",
                                 "Absolute nonlinear residual convergence tolerance (>= 0)"});
                rules.push_back({base + "linear/iterations", LiveConfigCategory::Steering, "integer", true, false, 1, false, 0, "iteration",
                                 "Maximum linear iterations per nonlinear correction (>= 1)"});
                rules.push_back({base + "linear/relativeTolerance", LiveConfigCategory::Steering, "number", true, false, 0, false, 0, "iteration",
                                 "Linear solver relative residual tolerance (>= 0)"});
                rules.push_back({base + "linear/absoluteTolerance", LiveConfigCategory::Steering, "number", true, false, 0, false, 0, "iteration",
                                 "Linear solver absolute residual tolerance (>= 0)"});
            }
            return rules;
        }

        void FlattenPatch(const Json &node, const std::string &prefix, std::vector<std::pair<std::string, Json>> &out)
        {
            if (node.is_object())
            {
                for (const auto &[key, value] : node.items())
                {
                    if (!key.empty() && key[0] == '/' && prefix.empty())
                        out.emplace_back(key, value);
                    else
                        FlattenPatch(value, prefix + "/" + key, out);
                }
            }
            else if (!prefix.empty())
                out.emplace_back(prefix, node);
        }

        std::string ResolvePointerAlias(const std::string &pointer, index domainCount)
        {
            if (pointer == "/time/maxWallSeconds")
                return "/time/wallSeconds";
            if (pointer == "/time/maxDt")
                return "/time/dt";
            if (domainCount == 1 && pointer.rfind("/implicit/", 0) == 0)
                return "/domains/0/settings" + pointer;
            return pointer;
        }

        bool IsValidIntegerJson(const Json &value, int64_t &out)
        {
            if (value.is_number_integer())
            {
                out = value.get<int64_t>();
                return true;
            }
            if (value.is_number_unsigned())
            {
                const auto u = value.get<uint64_t>();
                if (u > uint64_t(std::numeric_limits<int64_t>::max()))
                    return false;
                out = int64_t(u);
                return true;
            }
            return false;
        }
    }

    const char *ToString(ControlBoundary boundary) noexcept
    {
        switch (boundary)
        {
        case ControlBoundary::Auto:
            return "auto";
        case ControlBoundary::Step:
            return "step";
        case ControlBoundary::Iteration:
            return "iteration";
        }
        return "auto";
    }

    const char *ToString(LiveConfigCategory category) noexcept
    {
        switch (category)
        {
        case LiveConfigCategory::Output:
            return "output";
        case LiveConfigCategory::Steering:
            return "steering";
        }
        return "output";
    }

    std::optional<ControlBoundary> ParseControlBoundary(std::string_view text) noexcept
    {
        if (text == "auto")
            return ControlBoundary::Auto;
        if (text == "step")
            return ControlBoundary::Step;
        if (text == "iteration")
            return ControlBoundary::Iteration;
        return std::nullopt;
    }

    Json ControlUpdateRecord::ToJson() const
    {
        Json out{
            {"revision", revision},
            {"kind", kind},
            {"category", category},
            {"applyAt", applyAt},
            {"state", state},
            {"message", message}};
        if (!patch.is_null() && !patch.empty())
            out["patch"] = patch;
        if (state == "applied" || state == "rejected")
        {
            out["appliedStep"] = appliedStep;
            out["appliedStage"] = appliedStage;
            out["appliedIteration"] = appliedIteration;
        }
        return out;
    }

    RunControl::RunControl(const SimulationConfig &configuration, int rank)
        : _configuration(configuration.Root()),
          _rank(rank),
          _domainCount(index(configuration.Domains().size()))
    {
        _status = {
            {"state", "initializing"},
            {"rank", _rank},
            {"domains", _domainCount},
            {"step", uint64_t(0)},
            {"stage", -1},
            {"iteration", -1},
            {"time", 0.0},
            {"wallSeconds", 0.0},
            {"steady", configuration.Time().steady},
            {"stopRequested", false},
            {"appliedConfigRevision", uint64_t(0)}};
    }

    Json RunControl::MutableConfigSchema(index domainCount)
    {
        const auto rules = BuildRules(domainCount);
        Json outputFields = Json::array();
        Json steeringFields = Json::array();
        for (const auto &rule : rules)
        {
            Json item{
                {"pointer", rule.pointer},
                {"category", ToString(rule.category)},
                {"type", rule.type},
                {"defaultApplyAt", rule.defaultApplyAt},
                {"description", rule.description}};
            if (rule.hasMin)
            {
                if (rule.exclusiveMin)
                    item["exclusiveMinimum"] = rule.minValue;
                else
                    item["minimum"] = rule.minValue;
            }
            if (rule.hasMax)
                item["maximum"] = rule.maxValue;
            if (rule.category == LiveConfigCategory::Output)
                outputFields.push_back(std::move(item));
            else
                steeringFields.push_back(std::move(item));
        }
        return {
            {"categories", {{"output", outputFields}, {"steering", steeringFields}}},
            {"boundaries", Json::array({"iteration", "step"})}};
    }

    std::pair<bool, std::string> RunControl::ValidatePatch(const Json &currentConfig,
                                                           const Json &patch,
                                                           LiveConfigCategory &outCategory,
                                                           Json &outNormalizedPatch)
    {
        if (!patch.is_object() || patch.empty())
            return {false, "live config patch must be a non-empty JSON object"};
        if (!currentConfig.contains("domains") || !currentConfig["domains"].is_array())
            return {false, "current configuration lacks domains array"};
        const index domainCount = index(currentConfig["domains"].size());
        const auto rules = BuildRules(domainCount);
        std::map<std::string, const FieldRule *> ruleMap;
        for (const auto &rule : rules)
            ruleMap.emplace(rule.pointer, &rule);

        std::vector<std::pair<std::string, Json>> flattened;
        FlattenPatch(patch, "", flattened);
        if (flattened.empty())
            return {false, "live config patch contains no leaf parameters"};

        outCategory = LiveConfigCategory::Output;
        outNormalizedPatch = Json::object();

        for (const auto &[rawPointer, rawValue] : flattened)
        {
            const auto pointer = ResolvePointerAlias(rawPointer, domainCount);
            const auto it = ruleMap.find(pointer);
            if (it == ruleMap.end())
                return {false, "parameter is not live-mutable: " + rawPointer};
            const auto &rule = *it->second;
            if (rule.category == LiveConfigCategory::Steering)
                outCategory = LiveConfigCategory::Steering;

            if (rule.type == "integer")
            {
                int64_t v = 0;
                if (!IsValidIntegerJson(rawValue, v))
                    return {false, "parameter requires an integer value: " + pointer};
                if (rule.hasMin && (rule.exclusiveMin ? !(double(v) > rule.minValue) : double(v) < rule.minValue))
                    return {false, "parameter below minimum: " + pointer};
                if (rule.hasMax && double(v) > rule.maxValue)
                    return {false, "parameter above maximum: " + pointer};
                outNormalizedPatch[pointer] = v;
            }
            else
            {
                if (!rawValue.is_number())
                    return {false, "parameter requires a numeric value: " + pointer};
                const double v = rawValue.get<double>();
                if (!std::isfinite(v))
                    return {false, "parameter must be finite: " + pointer};
                if (rule.hasMin && (rule.exclusiveMin ? !(v > rule.minValue) : v < rule.minValue))
                    return {false, "parameter below minimum: " + pointer};
                if (rule.hasMax && v > rule.maxValue)
                    return {false, "parameter above maximum: " + pointer};
                outNormalizedPatch[pointer] = v;
            }
        }

        const Json candidate = ApplyNormalizedPatchToJson(currentConfig, outNormalizedPatch);
        const auto &timeJson = candidate.at("time");
        const bool steady = timeJson.value("steady", false);
        const std::string method = timeJson.value("method", std::string{"bdf1"});
        if (steady)
        {
            if (outNormalizedPatch.contains("/time/tEnd") && timeJson.at("tEnd").get<double>() > 0)
                return {false, "time.tEnd cannot be positive in steady mode"};
            if (outNormalizedPatch.contains("/time/cfl") && timeJson.at("cfl").get<double>() > 0)
                return {false, "time.cfl cannot be positive in steady mode"};
        }

        const auto &outputJson = candidate.at("output");
        const auto &impOut = outputJson.at("implicit");
        const int64_t impCkpt = impOut.at("checkpointInterval").get<int64_t>();
        const int64_t impVisRaw = impOut.at("visualizationInterval").get<int64_t>();
        const int64_t impVis = impVisRaw < 0 ? impCkpt : impVisRaw;
        if (impCkpt > 0 && domainCount > 1)
            return {false, "output.implicit.checkpointInterval requires a single domain"};
        if ((impCkpt > 0 || impVis > 0) && !steady && method == "ssprk3")
            return {false, "internal-iteration output requires an implicit time method"};

        for (index d = 0; d < domainCount; ++d)
        {
            const auto &imp = candidate["domains"][size_t(d)]["settings"]["implicit"];
            const int64_t iters = imp.at("iterations").get<int64_t>();
            const int64_t minIters = imp.at("minIterations").get<int64_t>();
            if (minIters > iters)
                return {false, "implicit.minIterations cannot exceed implicit.iterations on domain " + std::to_string(d)};
            const double pCfl = imp.at("pseudoCFL").get<double>();
            const double maxPCfl = imp.at("maxPseudoCFL").get<double>();
            if (maxPCfl < pCfl)
                return {false, "implicit.maxPseudoCFL cannot be smaller than implicit.pseudoCFL on domain " + std::to_string(d)};
        }

        return {true, ""};
    }

    Json RunControl::ApplyNormalizedPatchToJson(const Json &currentConfig,
                                                const Json &normalizedPatch)
    {
        Json updated = currentConfig;
        for (const auto &[pointer, value] : normalizedPatch.items())
            updated[Json::json_pointer(pointer)] = value;
        return updated;
    }

    void RunControl::RecordUpdateLocked(ControlUpdateRecord record)
    {
        _updates.push_back(std::move(record));
        if (_updates.size() > UpdateHistoryCapacity)
            _updates.pop_front();
    }

    void RunControl::TransitionUpdateLocked(uint64_t revision, const std::string &kind,
                                            const std::string &state, const std::string &message,
                                            uint64_t step, index stage, index iteration)
    {
        for (auto it = _updates.rbegin(); it != _updates.rend(); ++it)
            if (it->revision == revision && it->kind == kind)
            {
                it->state = state;
                it->message = message;
                it->appliedStep = step;
                it->appliedStage = stage;
                it->appliedIteration = iteration;
                return;
            }
        ControlUpdateRecord rec;
        rec.revision = revision;
        rec.kind = kind;
        rec.state = state;
        rec.message = message;
        rec.appliedStep = step;
        rec.appliedStage = stage;
        rec.appliedIteration = iteration;
        RecordUpdateLocked(std::move(rec));
    }

    void RunControl::EnqueueStop(ControlBoundary boundary)
    {
        // Escalate boundary urgency if a stop is already queued:
        // Iteration > Auto > Step when a user asks for a faster stop.
        const int current = _stopBoundary.load(std::memory_order_relaxed);
        const bool escalate = !_stop.load(std::memory_order_relaxed) ||
                              boundary == ControlBoundary::Iteration ||
                              (boundary == ControlBoundary::Auto && current == int(ControlBoundary::Step));
        if (escalate)
            _stopBoundary.store(int(boundary), std::memory_order_relaxed);
        _stop.store(true, std::memory_order_release);

        std::lock_guard<std::mutex> lock(_mutex);
        _status["stopRequested"] = true;
        _status["stopBoundary"] = ToString(ControlBoundary(_stopBoundary.load(std::memory_order_relaxed)));
        ControlUpdateRecord rec;
        rec.revision = _nextRevision++;
        rec.kind = "stop";
        rec.category = ToString(LiveConfigCategory::Steering);
        rec.applyAt = ToString(boundary);
        rec.state = "pending";
        rec.message = "graceful stop requested";
        RecordUpdateLocked(std::move(rec));
    }

    bool RunControl::StopRequested(ControlBoundary boundary, bool steady) const noexcept
    {
        if (!_stop.load(std::memory_order_acquire))
            return false;
        const auto requested = ControlBoundary(_stopBoundary.load(std::memory_order_relaxed));
        if (boundary == ControlBoundary::Step)
            return true;
        if (boundary == ControlBoundary::Iteration)
            return requested == ControlBoundary::Iteration ||
                   (requested == ControlBoundary::Auto && steady);
        return true;
    }

    std::pair<bool, ControlUpdateRecord> RunControl::EnqueueConfigPatch(
        const Json &patch, ControlBoundary applyAt)
    {
        if (applyAt == ControlBoundary::Auto)
            applyAt = ControlBoundary::Step;

        std::lock_guard<std::mutex> lock(_mutex);
        // Validate against effective config including already-queued patches
        // so sequential updates compose cleanly.
        Json effective = _configuration;
        for (const auto &pending : _pendingConfigs)
            effective = ApplyNormalizedPatchToJson(effective, pending.patch);

        LiveConfigCategory category = LiveConfigCategory::Output;
        Json normalized = Json::object();
        const auto [ok, error] = ValidatePatch(effective, patch, category, normalized);

        ControlUpdateRecord record;
        record.revision = _nextRevision++;
        record.kind = "config";
        record.category = ToString(category);
        record.applyAt = ToString(applyAt);
        record.patch = normalized.empty() ? patch : normalized;
        if (!ok)
        {
            record.state = "rejected";
            record.message = error;
            RecordUpdateLocked(record);
            return {false, record};
        }
        record.state = "pending";
        record.message = "queued for collective application at next " + record.applyAt + " boundary";
        _pendingConfigs.push_back({record.revision, applyAt, category, normalized});
        RecordUpdateLocked(record);
        return {true, record};
    }

    ControlUpdateRecord RunControl::EnqueueSnapshot(bool checkpoint, bool visualization,
                                                    ControlBoundary applyAt)
    {
        if (applyAt == ControlBoundary::Auto)
            applyAt = ControlBoundary::Step;
        std::lock_guard<std::mutex> lock(_mutex);
        ControlUpdateRecord record;
        record.revision = _nextRevision++;
        record.kind = "snapshot";
        record.category = ToString(LiveConfigCategory::Output);
        record.applyAt = ToString(applyAt);
        record.patch = {{"checkpoint", checkpoint}, {"visualization", visualization}};
        if (!checkpoint && !visualization)
        {
            record.state = "rejected";
            record.message = "snapshot request must enable checkpoint and/or visualization";
            RecordUpdateLocked(record);
            return record;
        }
        record.state = "pending";
        record.message = "queued snapshot";
        _pendingSnapshots.push_back({record.revision, applyAt, checkpoint, visualization});
        RecordUpdateLocked(record);
        return record;
    }

    std::vector<LiveConfigRequest> RunControl::TakePendingConfigPatches(ControlBoundary boundary)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        std::vector<LiveConfigRequest> taken;
        if (boundary == ControlBoundary::Step)
        {
            taken.assign(std::make_move_iterator(_pendingConfigs.begin()),
                         std::make_move_iterator(_pendingConfigs.end()));
            _pendingConfigs.clear();
            return taken;
        }
        std::deque<LiveConfigRequest> remaining;
        for (auto &req : _pendingConfigs)
        {
            if (req.applyAt == ControlBoundary::Iteration)
                taken.push_back(std::move(req));
            else
                remaining.push_back(std::move(req));
        }
        _pendingConfigs = std::move(remaining);
        return taken;
    }

    std::optional<LiveSnapshotRequest> RunControl::TakePendingSnapshot(ControlBoundary boundary)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        std::optional<LiveSnapshotRequest> merged;
        std::deque<LiveSnapshotRequest> remaining;
        for (const auto &req : _pendingSnapshots)
        {
            const bool due = boundary == ControlBoundary::Step || req.applyAt == ControlBoundary::Iteration;
            if (due)
            {
                if (!merged)
                    merged = req;
                else
                {
                    merged->id = req.id;
                    merged->checkpoint = merged->checkpoint || req.checkpoint;
                    merged->visualization = merged->visualization || req.visualization;
                }
            }
            else
                remaining.push_back(req);
        }
        _pendingSnapshots = std::move(remaining);
        return merged;
    }

    void RunControl::RecordAppliedConfig(uint64_t revision, const Json &newConfiguration,
                                         uint64_t step, index stage, index iteration)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _configuration = newConfiguration;
        _appliedConfigRevision = std::max(_appliedConfigRevision, revision);
        _status["appliedConfigRevision"] = _appliedConfigRevision;
        TransitionUpdateLocked(revision, "config", "applied", "applied collectively", step, stage, iteration);
    }

    void RunControl::RecordRejectedConfig(uint64_t revision, const std::string &reason,
                                          uint64_t step, index stage, index iteration)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        TransitionUpdateLocked(revision, "config", "rejected", reason, step, stage, iteration);
    }

    void RunControl::RecordAppliedSnapshot(uint64_t id, uint64_t step, index stage, index iteration)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        TransitionUpdateLocked(id, "snapshot", "applied", "snapshot written", step, stage, iteration);
    }

    void RunControl::RecordStopApplied(uint64_t step, index stage, index iteration,
                                       const std::string &boundaryName)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _status["state"] = "stopping";
        for (auto it = _updates.rbegin(); it != _updates.rend(); ++it)
            if (it->kind == "stop" && it->state == "pending")
            {
                it->state = "applied";
                it->message = "graceful stop triggered at " + boundaryName + " boundary";
                it->appliedStep = step;
                it->appliedStage = stage;
                it->appliedIteration = iteration;
            }
    }

    void RunControl::UpdateStatus(const Json &status)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (status.is_object())
            _status.update(status);
        _status["stopRequested"] = _stop.load(std::memory_order_relaxed);
        _status["pendingConfigCount"] = _pendingConfigs.size();
        _status["pendingSnapshotCount"] = _pendingSnapshots.size();
        _status["appliedConfigRevision"] = _appliedConfigRevision;
    }

    void RunControl::SetBoundaryMetadata(index domain, const Json &specs)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _boundarySpecs[std::to_string(domain)] = specs;
    }

    void RunControl::UpdateBoundaryLatest(index domain, int zone, const Json &row)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _boundaryLatest[std::to_string(domain)][std::to_string(zone)] = row;
    }

    Json RunControl::CurrentConfiguration() const
    {
        std::lock_guard<std::mutex> lock(_mutex);
        return _configuration;
    }

    Json RunControl::CurrentStatus() const
    {
        std::lock_guard<std::mutex> lock(_mutex);
        Json copy = _status;
        copy["pendingConfigCount"] = _pendingConfigs.size();
        copy["pendingSnapshotCount"] = _pendingSnapshots.size();
        copy["stopRequested"] = _stop.load(std::memory_order_relaxed);
        copy["stopBoundary"] = ToString(ControlBoundary(_stopBoundary.load(std::memory_order_relaxed)));
        copy["appliedConfigRevision"] = _appliedConfigRevision;
        return copy;
    }

    Json RunControl::CurrentBoundaries() const
    {
        std::lock_guard<std::mutex> lock(_mutex);
        return {{"domains", _boundarySpecs}, {"latest", _boundaryLatest}};
    }

    Json RunControl::RecentUpdates() const
    {
        std::lock_guard<std::mutex> lock(_mutex);
        Json arr = Json::array();
        for (const auto &rec : _updates)
            arr.push_back(rec.ToJson());
        return arr;
    }

    void RunControl::RetainHistory(index domain, const CsvRow &row)
    {
        std::lock_guard<std::mutex> lock(_historyMutex);
        HistoryEntry entry{_nextHistorySeq++, domain, row};
        _history.push_back(std::move(entry));
        if (_history.size() > HistoryCapacity)
            _history.pop_front();
    }

    Json RunControl::CurrentHistory(size_t limit) const
    {
        return CurrentHistory(limit, 0, std::nullopt);
    }

    Json RunControl::CurrentHistory(size_t limit, uint64_t afterSeq,
                                    std::optional<index> domainFilter) const
    {
        size_t retained = 0;
        uint64_t latestSeq = 0;
        std::vector<HistoryEntry> snapshot;
        {
            std::lock_guard<std::mutex> lock(_historyMutex);
            retained = _history.size();
            latestSeq = _history.empty() ? 0 : _history.back().seq;
            if (limit > 0)
            {
                for (const auto &entry : _history)
                {
                    if (entry.seq <= afterSeq)
                        continue;
                    if (domainFilter && entry.domain != *domainFilter)
                        continue;
                    snapshot.push_back(entry);
                }
                if (snapshot.size() > limit)
                    snapshot.erase(snapshot.begin(), snapshot.end() - std::ptrdiff_t(limit));
            }
        }
        Json records = Json::array();
        for (const auto &entry : snapshot)
        {
            Json values = Json::object();
            const auto &columns = entry.row.Columns();
            const auto &cells = entry.row.Cells();
            for (size_t i = 0; i < cells.size(); ++i)
                std::visit([&](const auto &value)
                           {
                    using T = std::decay_t<decltype(value)>;
                    if constexpr (!std::is_same_v<T, std::monostate>)
                        values[columns[i].name] = value; }, cells[i]);
            records.push_back({{"seq", entry.seq}, {"domain", entry.domain}, {"values", std::move(values)}});
        }
        return {
            {"rank", _rank},
            {"capacity", HistoryCapacity},
            {"retained", retained},
            {"latestSeq", latestSeq},
            {"records", std::move(records)}};
    }
}
