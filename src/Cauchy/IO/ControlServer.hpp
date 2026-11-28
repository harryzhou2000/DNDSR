#pragma once

#include "Cauchy/Config/SimulationConfig.hpp"
#include "Cauchy/IO/RunControl.hpp"

#include <memory>
#include <string>

namespace DNDS::Cauchy
{
    class OperationalLogger;

    /// Embedded rank-0 HTTP query and live steering server backed by cpp-httplib.
    /// Startup is non-blocking and non-fatal: if binding fails, the server logs a
    /// warning and leaves Running() == false without interrupting the simulation.
    class ControlServer
    {
        struct Data;
        std::unique_ptr<Data> _data;

    public:
        ControlServer(RunControl &control,
                      const ControlSettings &settings,
                      const std::string &prefix,
                      const std::string &runManifestPath,
                      int rank,
                      std::shared_ptr<OperationalLogger> logger = {});
        ~ControlServer();
        ControlServer(const ControlServer &) = delete;
        ControlServer &operator=(const ControlServer &) = delete;

        /// Attempt to bind and start the listener thread on rank 0.
        /// Returns true if listening, false if disabled, non-root, or bind failed.
        bool Start() noexcept;

        /// Stop the listener thread, clean up run-registry entry, and mark
        /// control file inactive. Safe to call multiple times.
        void Stop() noexcept;

        bool Running() const noexcept;
        std::string Host() const;
        int Port() const noexcept;
        std::string Url() const;
        std::string Token() const;
        std::string ControlFilePath() const;
        std::string RegistryFilePath() const;
        std::string FailureReason() const;
    };
}
