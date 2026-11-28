#include "Cauchy/IO/ControlServer.hpp"
#include "Cauchy/IO/ProgressOutput.hpp"
#include "Cauchy/IO/HistoryQuery.hpp"

#include <httplib.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <thread>

#ifdef _WIN32
#    include <process.h>
#else
#    include <sys/stat.h>
#    include <unistd.h>
#endif

namespace DNDS::Cauchy
{
    namespace
    {
        long CurrentPid() noexcept
        {
#ifdef _WIN32
            return long(_getpid());
#else
            return long(getpid());
#endif
        }

        std::string GenerateTokenHex(size_t bytes = 32)
        {
            std::random_device rd;
            std::ostringstream out;
            out << std::hex << std::setfill('0');
            for (size_t i = 0; i < bytes; i += 4)
            {
                const uint32_t word = rd();
                out << std::setw(8) << word;
            }
            return out.str().substr(0, bytes * 2);
        }

        std::string UtcTimestampIso()
        {
            const auto now = std::chrono::system_clock::now();
            const std::time_t tt = std::chrono::system_clock::to_time_t(now);
            std::tm tm{};
#ifdef _WIN32
            gmtime_s(&tm, &tt);
#else
            gmtime_r(&tt, &tm);
#endif
            char buf[32];
            std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
            return std::string(buf);
        }

        bool ConstantTimeEquals(std::string_view a, std::string_view b) noexcept
        {
            if (a.size() != b.size())
                return false;
            unsigned char diff = 0;
            for (size_t i = 0; i < a.size(); ++i)
                diff |= static_cast<unsigned char>(a[i] ^ b[i]);
            return diff == 0;
        }

        bool WritePrivateJsonFile(const std::filesystem::path &target, const Json &payload) noexcept
        {
            try
            {
                if (!target.parent_path().empty())
                    std::filesystem::create_directories(target.parent_path());
                const auto temp = std::filesystem::path(target.string() + ".tmp." + std::to_string(CurrentPid()));
                {
                    std::ofstream out(temp, std::ios::binary | std::ios::trunc);
                    if (!out)
                        return false;
                    out << payload.dump(2) << '\n';
                    out.flush();
                    if (!out)
                        return false;
                }
#ifndef _WIN32
                ::chmod(temp.c_str(), S_IRUSR | S_IWUSR);
#endif
                std::error_code ec;
                std::filesystem::rename(temp, target, ec);
                if (ec)
                {
                    std::filesystem::remove(temp, ec);
                    return false;
                }
                return true;
            }
            catch (...)
            {
                return false;
            }
        }

        std::filesystem::path ResolveRegistryDirectory(const ControlSettings &settings)
        {
            if (!settings.registryDir.empty())
                return std::filesystem::path(settings.registryDir);
            if (const char *envDir = std::getenv("DNDS_RUN_REGISTRY_DIR"); envDir && *envDir)
                return std::filesystem::path(envDir);
#ifdef _WIN32
            if (const char *profile = std::getenv("USERPROFILE"); profile && *profile)
                return std::filesystem::path(profile) / ".dndsr" / "runs";
#else
            if (const char *home = std::getenv("HOME"); home && *home)
                return std::filesystem::path(home) / ".dndsr" / "runs";
#endif
            return std::filesystem::temp_directory_path() / "dndsr-runs";
        }

        std::string MakeRunId(const std::string &prefix, int port)
        {
            std::string stem = std::filesystem::path(prefix).filename().string();
            if (stem.empty())
                stem = "cauchy";
            std::string safe;
            for (const unsigned char c : stem)
            {
                if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || c == '_')
                    safe.push_back(char(c));
                else
                    safe.push_back('_');
            }
            if (safe.size() > 32)
                safe.resize(32);
            return safe + "-p" + std::to_string(CurrentPid()) + "-" + std::to_string(port) + "-" + GenerateTokenHex(4);
        }

        std::string ExtractToken(const httplib::Request &req)
        {
            if (req.has_header("Authorization"))
            {
                const auto header = req.get_header_value("Authorization");
                static constexpr std::string_view Bearer = "Bearer ";
                if (header.size() > Bearer.size() && header.compare(0, Bearer.size(), Bearer) == 0)
                    return header.substr(Bearer.size());
            }
            if (req.has_header("X-Cauchy-Token"))
                return req.get_header_value("X-Cauchy-Token");
            return {};
        }

        void SendJson(httplib::Response &res, int status, const Json &body)
        {
            res.status = status;
            res.set_content(body.dump(), "application/json; charset=utf-8");
        }
    }

    struct ControlServer::Data
    {
        RunControl &control;
        HistoryQuery historyQuery;
        ControlSettings settings;
        std::string prefix;
        std::string runManifestPath;
        int rank = 0;
        std::shared_ptr<OperationalLogger> logger;

        std::unique_ptr<httplib::Server> server;
        std::thread thread;
        std::atomic<bool> running{false};

        std::string runId;
        std::string token;
        std::string host;
        int port = 0;
        std::string controlFilePath;
        std::string registryFilePath;
        std::string failureReason;
        std::string startedAt;

        Data(RunControl &controlIn, ControlSettings settingsIn,
             std::string prefixIn, std::string manifestIn,
             int rankIn, std::shared_ptr<OperationalLogger> loggerIn)
            : control(controlIn),
              settings(std::move(settingsIn)),
              prefix(std::move(prefixIn)),
              runManifestPath(std::move(manifestIn)),
              rank(rankIn),
              logger(std::move(loggerIn)) {}

        void LogInfo(const std::string &msg) const
        {
            if (logger)
                logger->Info(msg);
            else
                std::cerr << msg << '\n';
        }

        void LogWarn(const std::string &msg) const
        {
            if (logger)
                logger->Warn(msg);
            else
                std::cerr << "WARNING: " << msg << '\n';
        }

        bool Authorize(const httplib::Request &req, httplib::Response &res) const
        {
            const auto supplied = ExtractToken(req);
            if (!ConstantTimeEquals(supplied, token))
            {
                SendJson(res, 401, {{"error", "unauthorized"}, {"message", "valid per-run bearer token required"}});
                return false;
            }
            return true;
        }

        void ExclusiveSocketOptions(socket_t sock)
        {
            // Never set SO_REUSEPORT: concurrent Cauchy runs must bind distinct
            // exclusive ports rather than sharing one port via kernel load-balancing.
            int opt = 1;
#ifdef _WIN32
            setsockopt(sock, SOL_SOCKET, SO_EXCLUSIVEADDRUSE,
                       reinterpret_cast<const char *>(&opt), sizeof(opt));
#else
            setsockopt(sock, SOL_SOCKET, SO_REUSEADDR,
                       reinterpret_cast<const void *>(&opt), sizeof(opt));
#endif
        }

        void RegisterRoutes()
        {
            server->Get("/api/v1/manifest", [this](const httplib::Request &req, httplib::Response &res)
                        {
                if (!Authorize(req, res)) return;
                try { std::ifstream in(runManifestPath); Json manifest; in >> manifest; SendJson(res, 200, manifest); }
                catch (...) { SendJson(res, 404, {{"error", "manifest_unavailable"}}); } });
            server->Get("/api/v1/history/query", [this](const httplib::Request &req, httplib::Response &res)
                        {
                if (!Authorize(req, res)) return;
                try
                {
                    const auto domText = req.has_param("domain") ? req.get_param_value("domain") : "0";
                    if (domText.empty() || domText.find_first_not_of("0123456789") != std::string::npos) throw std::invalid_argument("invalid domain");
                    const auto domain = std::stoll(domText);
                    std::ifstream in(runManifestPath); Json manifest; in >> manifest;
                    std::string path;
                    for (const auto &d : manifest.at("domains"))
                    {
                        if (d.at("index").get<int64_t>() != domain) continue;
                        if (!req.has_param("boundary")) path = d.at("history").get<std::string>();
                        else for (const auto &b : d.value("boundaryIntegrations", Json::array()))
                            if (b.at("canonicalName") == req.get_param_value("boundary")) path = b.at("csv").get<std::string>();
                    }
                    if (path.empty()) { SendJson(res, 404, {{"error", "history_not_found"}}); return; }
                    std::map<std::string, std::string> args;
                    for (const auto &p : req.params) args[p.first] = p.second;
                    SendJson(res, 200, historyQuery.Query(path, args));
                }
                catch (const std::invalid_argument &e) { SendJson(res, 400, {{"error", "invalid_query"}, {"message", e.what()}}); }
                catch (const std::out_of_range &e) { SendJson(res, 400, {{"error", "invalid_query"}, {"message", e.what()}}); }
                catch (const std::exception &e) { SendJson(res, 503, {{"error", "history_unavailable"}, {"message", e.what()}}); } });
            server->set_read_timeout(5, 0);
            server->set_write_timeout(5, 0);
            server->set_idle_interval(0, 50000); // 50ms for responsive Stop()

            server->Get("/api/v1/status", [this](const httplib::Request &req, httplib::Response &res)
                        {
                if (!Authorize(req, res))
                    return;
                Json payload = control.CurrentStatus();
                payload["runId"] = runId;
                payload["pid"] = CurrentPid();
                payload["endpoint"] = {
                    {"host", host},
                    {"port", port},
                    {"url", "http://" + host + ":" + std::to_string(port)},
                    {"controlFile", controlFilePath}};
                payload["prefix"] = prefix;
                payload["runManifest"] = runManifestPath;
                payload["startedAt"] = startedAt;
                payload["updates"] = control.RecentUpdates();
                SendJson(res, 200, payload); });

            server->Get("/api/v1/config", [this](const httplib::Request &req, httplib::Response &res)
                        {
                if (!Authorize(req, res))
                    return;
                SendJson(res, 200, {
                                       {"configuration", control.CurrentConfiguration()},
                                       {"mutableSchema", control.MutableSchema()},
                                       {"updates", control.RecentUpdates()}}); });

            server->Get("/api/v1/history", [this](const httplib::Request &req, httplib::Response &res)
                        {
                if (!Authorize(req, res))
                    return;
                size_t limit = RunControl::HistoryCapacity;
                uint64_t afterSeq = 0;
                std::optional<index> domainFilter;
                try
                {
                    if (req.has_param("limit"))
                        limit = size_t(std::stoull(req.get_param_value("limit")));
                    if (req.has_param("after"))
                        afterSeq = std::stoull(req.get_param_value("after"));
                    if (req.has_param("domain"))
                    {
                        const auto dom = req.get_param_value("domain");
                        if (!dom.empty() && dom != "all")
                            domainFilter = index(std::stoll(dom));
                    }
                }
                catch (...)
                {
                    SendJson(res, 400, {{"error", "invalid_query"}, {"message", "invalid limit, after, or domain query parameter"}});
                    return;
                }
                SendJson(res, 200, control.CurrentHistory(limit, afterSeq, domainFilter)); });

            server->Get("/api/v1/boundaries", [this](const httplib::Request &req, httplib::Response &res)
                        {
                if (!Authorize(req, res))
                    return;
                SendJson(res, 200, control.CurrentBoundaries()); });

            server->Post("/api/v1/stop", [this](const httplib::Request &req, httplib::Response &res)
                         {
                if (!Authorize(req, res))
                    return;
                ControlBoundary boundary = ControlBoundary::Auto;
                if (!req.body.empty())
                {
                    try
                    {
                        const auto body = Json::parse(req.body);
                        if (body.is_object() && body.contains("applyAt"))
                        {
                            const auto parsed = ParseControlBoundary(body.at("applyAt").get<std::string>());
                            if (!parsed)
                            {
                                SendJson(res, 400, {{"error", "invalid_boundary"}, {"message", "applyAt must be auto, iteration, or step"}});
                                return;
                            }
                            boundary = *parsed;
                        }
                    }
                    catch (const std::exception &ex)
                    {
                        SendJson(res, 400, {{"error", "invalid_json"}, {"message", ex.what()}});
                        return;
                    }
                }
                control.EnqueueStop(boundary);
                SendJson(res, 202, {
                                       {"accepted", true},
                                       {"applyAt", ToString(boundary)},
                                       {"stopBoundary", ToString(control.StopBoundary())}}); });

            server->Post("/api/v1/snapshot", [this](const httplib::Request &req, httplib::Response &res)
                         {
                if (!Authorize(req, res))
                    return;
                bool checkpoint = true;
                bool visualization = true;
                ControlBoundary boundary = ControlBoundary::Step;
                if (!req.body.empty())
                {
                    try
                    {
                        const auto body = Json::parse(req.body);
                        if (!body.is_object())
                        {
                            SendJson(res, 400, {{"error", "invalid_json"}, {"message", "body must be a JSON object"}});
                            return;
                        }
                        if (body.contains("checkpoint"))
                            checkpoint = body.at("checkpoint").get<bool>();
                        if (body.contains("visualization"))
                            visualization = body.at("visualization").get<bool>();
                        if (body.contains("applyAt"))
                        {
                            const auto parsed = ParseControlBoundary(body.at("applyAt").get<std::string>());
                            if (!parsed)
                            {
                                SendJson(res, 400, {{"error", "invalid_boundary"}, {"message", "applyAt must be iteration or step"}});
                                return;
                            }
                            boundary = *parsed;
                        }
                    }
                    catch (const std::exception &ex)
                    {
                        SendJson(res, 400, {{"error", "invalid_json"}, {"message", ex.what()}});
                        return;
                    }
                }
                const auto record = control.EnqueueSnapshot(checkpoint, visualization, boundary);
                SendJson(res, record.state == "rejected" ? 400 : 202, record.ToJson()); });

            const auto handleConfigPatch = [this](const httplib::Request &req, httplib::Response &res)
            {
                if (!Authorize(req, res))
                    return;
                try
                {
                    const auto body = Json::parse(req.body);
                    if (!body.is_object())
                    {
                        SendJson(res, 400, {{"error", "invalid_json"}, {"message", "body must be a JSON object"}});
                        return;
                    }
                    ControlBoundary boundary = ControlBoundary::Step;
                    if (body.contains("applyAt"))
                    {
                        const auto parsed = ParseControlBoundary(body.at("applyAt").get<std::string>());
                        if (!parsed || *parsed == ControlBoundary::Auto)
                        {
                            SendJson(res, 400, {{"error", "invalid_boundary"}, {"message", "applyAt must be iteration or step"}});
                            return;
                        }
                        boundary = *parsed;
                    }
                    const Json patch = body.contains("patch") ? body.at("patch") : body;
                    Json cleanPatch = patch;
                    if (!body.contains("patch") && cleanPatch.is_object())
                        cleanPatch.erase("applyAt");
                    const auto [ok, record] = control.EnqueueConfigPatch(cleanPatch, boundary);
                    SendJson(res, ok ? 202 : 400, record.ToJson());
                }
                catch (const std::exception &ex)
                {
                    SendJson(res, 400, {{"error", "invalid_json"}, {"message", ex.what()}});
                }
            };
            server->Patch("/api/v1/config", handleConfigPatch);
            server->Post("/api/v1/config", handleConfigPatch);
        }

        Json BuildControlFilePayload(bool active) const
        {
            std::error_code ec;
            const auto cwd = std::filesystem::current_path(ec).string();
            const auto absPrefix = prefix.empty() ? std::string{} : std::filesystem::absolute(prefix, ec).string();
            const auto absManifest = runManifestPath.empty() ? std::string{} : std::filesystem::absolute(runManifestPath, ec).string();
            Json payload{
                {"schema", "dndsr.cauchy.control.v1"},
                {"active", active},
                {"runId", runId},
                {"pid", CurrentPid()},
                {"host", host},
                {"port", port},
                {"url", "http://" + host + ":" + std::to_string(port)},
                {"prefix", absPrefix.empty() ? prefix : absPrefix},
                {"runManifest", absManifest.empty() ? runManifestPath : absManifest},
                {"workingDirectory", cwd},
                {"startedAt", startedAt}};
            if (active)
                payload["token"] = token;
            return payload;
        }
    };

    ControlServer::ControlServer(RunControl &control,
                                 const ControlSettings &settings,
                                 const std::string &prefix,
                                 const std::string &runManifestPath,
                                 int rank,
                                 std::shared_ptr<OperationalLogger> logger)
        : _data(std::make_unique<Data>(control, settings, prefix, runManifestPath, rank, std::move(logger))) {}

    ControlServer::~ControlServer()
    {
        Stop();
    }

    bool ControlServer::Start() noexcept
    {
        if (!_data || _data->rank != 0 || !_data->settings.enabled || _data->running.load())
            return _data && _data->running.load();
        try
        {
            _data->host = _data->settings.host;
            _data->token = GenerateTokenHex(32);
            _data->startedAt = UtcTimestampIso();

            // cpp-httplib permanently decommissions a Server after a failed bind,
            // so construct a fresh Server per candidate port and register routes once bound.
            const auto tryBind = [&](int candidate) -> int
            {
                auto trial = std::make_unique<httplib::Server>();
                trial->set_socket_options([this](socket_t sock)
                                          { _data->ExclusiveSocketOptions(sock); });
                const int bound = candidate > 0
                                      ? (trial->bind_to_port(_data->host, candidate) ? candidate : -1)
                                      : trial->bind_to_any_port(_data->host);
                if (bound > 0)
                    _data->server = std::move(trial);
                return bound;
            };

            int boundPort = -1;
            const int startPort = _data->settings.port;
            const int scanCount = std::max(1, _data->settings.portScanCount);
            if (startPort > 0)
            {
                const int endPort = std::min(65535, startPort + scanCount - 1);
                for (int candidate = startPort; candidate <= endPort; ++candidate)
                {
                    boundPort = tryBind(candidate);
                    if (boundPort > 0)
                        break;
                }
            }
            if (boundPort <= 0)
                boundPort = tryBind(0);

            if (boundPort <= 0 || !_data->server)
            {
                _data->failureReason = "unable to bind HTTP control server on host " + _data->host;
                _data->LogWarn("Cauchy HTTP control disabled: " + _data->failureReason);
                _data->server.reset();
                return false;
            }
            _data->RegisterRoutes();

            _data->port = boundPort;
            _data->runId = MakeRunId(_data->prefix, boundPort);
            _data->running.store(true);
            _data->thread = std::thread([data = _data.get()]
                                        {
                try
                {
                    data->server->listen_after_bind();
                }
                catch (...)
                {
                    // Non-fatal: never terminate the solver if the control listener exits.
                    data->running.store(false);
                }
                data->running.store(false); });

            // Wait briefly for server socket loop to enter listening state.
            for (int wait = 0; wait < 200 && !_data->server->is_running(); ++wait)
                std::this_thread::sleep_for(std::chrono::milliseconds(5));

            const auto payload = _data->BuildControlFilePayload(true);
            if (!_data->prefix.empty())
            {
                _data->controlFilePath = _data->prefix + "-control.json";
                if (!WritePrivateJsonFile(_data->controlFilePath, payload))
                    _data->LogWarn("Cauchy HTTP control: failed to write " + _data->controlFilePath);
            }
            if (_data->settings.registerRun)
            {
                const auto regDir = ResolveRegistryDirectory(_data->settings);
                _data->registryFilePath = (regDir / (_data->runId + ".json")).string();
                if (!WritePrivateJsonFile(_data->registryFilePath, payload))
                {
                    _data->LogWarn("Cauchy HTTP control: failed to write registry entry " + _data->registryFilePath);
                    _data->registryFilePath.clear();
                }
            }
            _data->LogInfo("Cauchy HTTP control listening on http://" + _data->host + ":" +
                           std::to_string(_data->port) +
                           (_data->controlFilePath.empty() ? "" : " (control file: " + _data->controlFilePath + ")"));
            return true;
        }
        catch (const std::exception &ex)
        {
            _data->failureReason = ex.what();
            _data->LogWarn("Cauchy HTTP control failed to start (continuing without HTTP): " + _data->failureReason);
            _data->server.reset();
            _data->running.store(false);
            return false;
        }
        catch (...)
        {
            _data->failureReason = "unknown startup error";
            _data->LogWarn("Cauchy HTTP control failed to start (continuing without HTTP)");
            _data->server.reset();
            _data->running.store(false);
            return false;
        }
    }

    void ControlServer::Stop() noexcept
    {
        if (!_data)
            return;
        try
        {
            if (_data->server)
                _data->server->stop();
            if (_data->thread.joinable())
                _data->thread.join();
            _data->running.store(false);
            _data->server.reset();

            if (!_data->registryFilePath.empty())
            {
                std::error_code ec;
                std::filesystem::remove(_data->registryFilePath, ec);
                _data->registryFilePath.clear();
            }
            if (!_data->controlFilePath.empty() && _data->port > 0)
                WritePrivateJsonFile(_data->controlFilePath, _data->BuildControlFilePayload(false));
        }
        catch (...)
        {
            // Never throw from Stop / destructor.
            _data->running.store(false);
        }
    }

    bool ControlServer::Running() const noexcept { return _data && _data->running.load(); }
    std::string ControlServer::Host() const { return _data ? _data->host : std::string{}; }
    int ControlServer::Port() const noexcept { return _data ? _data->port : 0; }
    std::string ControlServer::Url() const
    {
        return Running() ? "http://" + _data->host + ":" + std::to_string(_data->port) : std::string{};
    }
    std::string ControlServer::Token() const { return _data ? _data->token : std::string{}; }
    std::string ControlServer::ControlFilePath() const { return _data ? _data->controlFilePath : std::string{}; }
    std::string ControlServer::RegistryFilePath() const { return _data ? _data->registryFilePath : std::string{}; }
    std::string ControlServer::FailureReason() const { return _data ? _data->failureReason : std::string{}; }
}
