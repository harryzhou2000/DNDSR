#include "Cauchy/Discretization/FiniteVolumeViews.hpp"

namespace DNDS::Cauchy
{
    RHSLocalDiagnostics::RHSLocalDiagnostics(std::vector<int> zones)
        : _zones(std::move(zones)), _forces(_zones.size())
    {
        DNDS_check_throw_info(std::is_sorted(_zones.begin(), _zones.end()) &&
                                  std::adjacent_find(_zones.begin(), _zones.end()) == _zones.end(),
                              "RHS diagnostic zones must be sorted and unique");
        Reset();
    }
    void RHSLocalDiagnostics::Reset()
    {
        _faceFallbacks = 0;
        std::fill(_forces.begin(), _forces.end(), Vector3{});
    }
    void RHSLocalDiagnostics::Merge(const RHSKernelDiagnostics &input)
    {
        // Standalone kernel users commonly need only the fallback count. Let an
        // empty aggregate adopt the immutable slot extent on first merge; named
        // BoundaryForce queries still require construction with the zone table.
        if (_zones.empty() && _forces.empty())
            _forces.resize(input._forces.size(), Vector3{});
        DNDS_check_throw_info(input._forces.size() == _forces.size(), "RHS diagnostic force-slot layout differs");
        _faceFallbacks += input._faceFallbacks;
        for (size_t slot = 0; slot < _forces.size(); ++slot)
            for (int d = 0; d < 3; ++d)
                _forces[slot][d] += input._forces[slot][d];
    }
    Vector3 RHSLocalDiagnostics::BoundaryForce(const std::vector<int> &zones) const
    {
        Vector3 force{};
        for (int zone : zones)
        {
            const auto found = std::lower_bound(_zones.begin(), _zones.end(), zone);
            DNDS_check_throw_info(found != _zones.end() && *found == zone, "force zone has no boundary condition");
            const auto slot = size_t(found - _zones.begin());
            for (int d = 0; d < 3; ++d)
                force[d] += _forces[slot][d];
        }
        return force;
    }
}

namespace DNDS::Cauchy::EquationFVDetail
{
    Runtime::MetadataBuffer<EquationBlockReadView> BlockViews(const EquationDofs &state, const EquationDofs::ReadScope &scope)
    {
        return Runtime::MetadataBuffer<EquationBlockReadView>::Generate(
            state.BlockNumber(), [&](index iBlock)
            { return scope.Block(iBlock); });
    }
}
