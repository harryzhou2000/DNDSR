#pragma once

#include "Cauchy/Core/StateEpoch.hpp"
#include "Cauchy/Config/SimulationConfig.hpp"
#include "Cauchy/State/EquationDofs.hpp"
#include "DNDS/Core/OptionalRef.hpp"

namespace DNDS::Cauchy
{
    class Domain;

    /// Managed immutable view of one accepted domain generation.
    ///
    /// Construction acquires owned read leases for every retained payload, so
    /// means, optional reconstruction, limited-VR history and BDF history
    /// cannot change independently while an output/checkpoint consumer runs.
    /// The snapshot is zero-copy and copyable by shared ownership. It contains
    /// no engine, writer or mutable solver facade; release it before committing
    /// a new accepted state.
    class DomainStateSnapshot
    {
        struct Data;
        std::shared_ptr<const Data> _data;
        DomainStateSnapshot(const Domain &domain, bool reconstruction);
        friend class Domain;

    public:
        const std::string &Name() const;
        const EquationDofs &Means() const;
        OptionalRef<const EquationDofs> Reconstruction() const;
        OptionalRef<const Runtime::Array<real>> LimitedVRHistory() const;
        OptionalRef<const EquationDofs> PreviousState() const;
        real Time() const;
        real LastAcceptedDt() const;
        real PreviousDt() const;
        const StateEpoch &Epoch() const;
        const std::string &TimeMethod() const;
        const Json &NonlinearState() const;
        const Json &UpdatePolicy() const;
        bool SourceStrang() const;
    };
}
