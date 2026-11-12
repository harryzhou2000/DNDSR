#include "Cauchy/Core/DomainPlacement.hpp"
#include "DNDS/MPI/Failure.hpp"
#include <nlohmann/json.hpp>
#include <set>
#include <climits>
#include <algorithm>

namespace DNDS::Cauchy
{
    struct DomainPlacement::State
    {
        std::shared_ptr<Runtime::CommunicationContext> control;
        std::vector<DomainRankGroup> groups;
        std::vector<std::shared_ptr<Runtime::CommunicationContext>> contexts;
        int rank = 0, size = 0;
    };
    DomainPlacement::DomainPlacement(MPI_Comm simulation, std::vector<DomainRankGroup> groups)
    {
        DNDS_check_throw_info(simulation != MPI_COMM_NULL, "Cauchy needs a simulation communicator");
        auto state = std::make_shared<State>();
        state->control = Runtime::CommunicationContext::Duplicate(simulation);
        MPI_Comm_rank(simulation, &state->rank);
        MPI_Comm_size(simulation, &state->size);
        int valid = !groups.empty();
        std::set<std::string> names;
        nlohmann::ordered_json description = nlohmann::ordered_json::array();
        for (const auto &group : groups)
        {
            valid &= !group.name.empty() && names.insert(group.name).second && !group.ranks.empty();
            std::set<int> seen;
            for (auto rank : group.ranks)
                valid &= rank >= 0 && rank < state->size && seen.insert(rank).second;
            description.push_back({group.name, group.ranks});
        }
        const auto serialized = description.dump();
        int length = state->rank == 0 && serialized.size() <= INT_MAX ? int(serialized.size()) : 0;
        MPI_Bcast(&length, 1, MPI_INT, 0, simulation);
        std::string reference(size_t(length), '\0');
        if (state->rank == 0)
            reference = serialized;
        MPI_Bcast(reference.data(), length, MPI_CHAR, 0, simulation);
        valid &= serialized == reference && length > 0;
        valid = MPI::AllreduceValue<MPI::Reduction::Min>(valid, simulation);
        MPI::CheckPreflight(valid, "Cauchy domain groups are invalid or differ across simulation ranks");
        for (const auto &group : groups)
        {
            const auto found = std::find(group.ranks.begin(), group.ranks.end(), state->rank);
            const auto member = found != group.ranks.end();
            MPI_Comm domain = MPI_COMM_NULL;
            MPI_Comm_split(simulation, member ? 0 : MPI_UNDEFINED, member ? int(found - group.ranks.begin()) : 0, &domain);
            state->contexts.push_back(member ? Runtime::CommunicationContext::Adopt(domain) : nullptr);
        }
        state->groups = std::move(groups);
        _state = std::move(state);
    }
    MPI_Comm DomainPlacement::Control() const { return _state->control->Communicator(); }
    std::shared_ptr<Runtime::CommunicationContext> DomainPlacement::ControlContext() const { return _state->control; }
    int DomainPlacement::Rank() const { return _state->rank; }
    int DomainPlacement::Size() const { return _state->size; }
    index DomainPlacement::DomainNumber() const { return index(_state->groups.size()); }
    const DomainRankGroup &DomainPlacement::Group(index domain) const
    {
        DNDS_check_throw_info(domain >= 0 && domain < DomainNumber(), "Cauchy domain index out of range");
        return _state->groups[domain];
    }
    int DomainPlacement::DomainRank(index domain) const
    {
        const auto &ranks = Group(domain).ranks;
        const auto found = std::find(ranks.begin(), ranks.end(), Rank());
        return found == ranks.end() ? -1 : int(found - ranks.begin());
    }
    int DomainPlacement::SimulationRank(index domain, int rank) const
    {
        const auto &ranks = Group(domain).ranks;
        DNDS_check_throw_info(rank >= 0 && rank < int(ranks.size()), "Cauchy domain rank out of range");
        return ranks[rank];
    }
    std::shared_ptr<Runtime::CommunicationContext> DomainPlacement::Context(index domain) const
    {
        Group(domain);
        DNDS_check_throw_info(_state->contexts[domain], "rank is not a member of this Cauchy domain");
        return _state->contexts[domain];
    }
    std::vector<index> DomainPlacement::LocalDomains() const
    {
        std::vector<index> result;
        for (index domain = 0; domain < DomainNumber(); ++domain)
            if (_state->contexts[domain])
                result.push_back(domain);
        return result;
    }
}
