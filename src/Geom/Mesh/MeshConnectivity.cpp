#include "MeshConnectivity.hpp"

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <fmt/core.h>

namespace DNDS::Geom
{
    // -----------------------------------------------------------------
    // adjKindName
    // -----------------------------------------------------------------

    std::vector<EntityKind> AdjKind::Path() const
    {
        if (!path.empty())
            return path;
        return isIntraLevel() ? std::vector<EntityKind>{from, via, to} : std::vector<EntityKind>{from, to};
    }

    std::string AdjKind::Signature() const
    {
        std::string result = std::to_string(int(from)) + ":" + std::to_string(int(to)) + ":";
        if (IsLegacy())
            return result + "legacy:" + (isIntraLevel() ? std::to_string(int(via)) : "direct");
        for (auto kind : path)
            result += std::to_string(int(kind)) + ",";
        result += ":" + std::to_string(int(selection)) + ":" + std::to_string(removeSelf) + ":" +
                  std::to_string(int(order)) + ":" + std::to_string(int(periodic)) + ":";
        result += std::to_string(semanticName.size()) + ":" + semanticName + ":" + std::to_string(semanticVersion) + ":";
        return result + std::to_string(provenance.size()) + ":" + provenance;
    }

    uint64_t AdjKind::DescriptorId() const
    {
        uint64_t value = 14695981039346656037ULL;
        for (unsigned char byte : Signature())
        {
            value ^= byte;
            value *= 1099511628211ULL;
        }
        return value;
    }

    void AdjKind::Validate() const
    {
        auto valid = [](EntityKind kind)
        { return kind >= EntityKind::Cell && kind < EntityKind::NUM_KINDS; };
        DNDS_check_throw_info(valid(from) && valid(to) && valid(via), "invalid relation entity kind");
        if (IsLegacy())
        {
            DNDS_check_throw_info(path.empty() && !removeSelf && order == RelationOrder::Legacy &&
                                      periodic == RelationPeriodicPolicy::Legacy && semanticName.empty() &&
                                      semanticVersion == 0 && provenance.empty(),
                                  "legacy relation adapter cannot hide native metadata");
            return;
        }
        DNDS_check_throw_info(path.size() >= 2 && path.front() == from && path.back() == to &&
                                  std::all_of(path.begin(), path.end(), valid),
                              "relation path differs from endpoints");
        DNDS_check_throw_info(selection >= RelationSelection::Incidence && selection <= RelationSelection::Custom &&
                                  order >= RelationOrder::Topological && order <= RelationOrder::SortedUnique &&
                                  periodic >= RelationPeriodicPolicy::EntityIdentity && periodic <= RelationPeriodicPolicy::FrameMatched,
                              "invalid relation selection/order/periodic policy");
        DNDS_check_throw_info((selection != RelationSelection::Custom && selection != RelationSelection::Exact) ||
                                  (!semanticName.empty() && semanticVersion > 0),
                              "custom/exact selection requires a stable semantic name and positive version");
    }

    AdjKind AdjKind::Composition(const AdjKind &ab, const AdjKind &bc, RelationSelection selection_,
                                 bool removeSelf_, std::string name, uint32_t version, RelationPeriodicPolicy periodic_)
    {
        ab.Validate();
        bc.Validate();
        DNDS_check_throw_info(ab.to == bc.from, "relation descriptor composition endpoint mismatch");
        AdjKind result(ab.from, bc.to, ab.to);
        result.path = ab.Path();
        const auto tail = bc.Path();
        result.path.insert(result.path.end(), tail.begin() + 1, tail.end());
        result.selection = selection_;
        result.removeSelf = removeSelf_;
        result.order = RelationOrder::SortedUnique;
        result.periodic = periodic_;
        if (name.empty() && (selection_ == RelationSelection::Any || selection_ == RelationSelection::All))
        {
            name = selection_ == RelationSelection::Any ? "geom.compose.any" : "geom.compose.all";
            version = 1;
        }
        result.semanticName = std::move(name);
        result.semanticVersion = version;
        const auto first = ab.Signature(), second = bc.Signature();
        result.provenance = "compose:" + std::to_string(first.size()) + ":" + first + second;
        result.Validate();
        return result;
    }

    AdjKind AdjKind::Inverted() const
    {
        Validate();
        // Inverting a legacy ordered cone produces its historical support
        // slot. Inverting a support produces sorted rows, so it must not claim
        // the element-topological ordering promised by the reverse cone slot.
        const bool legacyCone = IsLegacy() && !isIntraLevel() &&
                                ((from == EntityKind::Cell && (to == EntityKind::Face || to == EntityKind::Edge || to == EntityKind::Node)) ||
                                 (from == EntityKind::Face && (to == EntityKind::Edge || to == EntityKind::Node)) ||
                                 ((from == EntityKind::Edge || from == EntityKind::Bnd) && to == EntityKind::Node));
        if (legacyCone)
            return {to, from};
        AdjKind result = *this;
        result.from = to;
        result.to = from;
        result.path = Path();
        std::reverse(result.path.begin(), result.path.end());
        if (IsLegacy())
        {
            result.selection = isIntraLevel() ? RelationSelection::Any : RelationSelection::Incidence;
            result.periodic = RelationPeriodicPolicy::EntityIdentity;
            result.semanticName = "geom.inverse.legacy";
            result.semanticVersion = 1;
        }
        result.order = RelationOrder::SortedUnique;
        result.provenance = "inverse:" + Signature();
        result.Validate();
        return result;
    }

    std::string adjKindName(const AdjKind &kind)
    {
        if (!kind.IsLegacy())
            return fmt::format("{}2{}[{}]", entityKindName(kind.from), entityKindName(kind.to), kind.Signature());
        if (kind.isDirect())
            return fmt::format("{}2{}", entityKindName(kind.from), entityKindName(kind.to));
        return fmt::format("{}2{}({})", entityKindName(kind.from), entityKindName(kind.to), entityKindName(kind.via));
    }

    // -----------------------------------------------------------------
    // Adjacency registry
    // -----------------------------------------------------------------

    void MeshConnectivity::registerAdj(AdjKind kind, ssp<AdjVariant> adjPtr)
    {
        kind.Validate();
        adjRegistry[kind] = std::move(adjPtr);
    }

    void MeshConnectivity::registerGlobalMapping(EntityKind kind, const ssp<GlobalOffsetsMapping> &gm)
    {
        globalMappings[kind] = gm;
    }

    ssp<AdjVariant> MeshConnectivity::resolveAdj(AdjKind kind) const
    {
        auto it = adjRegistry.find(kind);
        return (it != adjRegistry.end()) ? it->second : nullptr;
    }

    const ssp<GlobalOffsetsMapping> &MeshConnectivity::getGlobalMapping(EntityKind kind) const
    {
        auto it = globalMappings.find(kind);
        if (it != globalMappings.end())
            return it->second;
        static const ssp<GlobalOffsetsMapping> null_mapping{};
        return null_mapping;
    }

    bool MeshConnectivity::hasAdj(AdjKind kind) const
    {
        return adjRegistry.find(kind) != adjRegistry.end();
    }

    // -----------------------------------------------------------------
    // Cone management
    // -----------------------------------------------------------------

    ConeAdj &MeshConnectivity::addCone(int fromDepth, int toDepth)
    {
        DNDS_assert_info(!hasCone(fromDepth, toDepth),
                         fmt::format("Cone ({}, {}) already exists", fromDepth, toDepth));
        cones.push_back(ConeAdj{fromDepth, toDepth, makeAdjVariant<tAdjPair>(), {}});
        return cones.back();
    }

    ConeAdj *MeshConnectivity::findCone(int fromDepth, int toDepth)
    {
        for (auto &c : cones)
            if (c.fromDepth == fromDepth && c.toDepth == toDepth)
                return &c;
        return nullptr;
    }

    const ConeAdj *MeshConnectivity::findCone(int fromDepth, int toDepth) const
    {
        for (const auto &c : cones)
            if (c.fromDepth == fromDepth && c.toDepth == toDepth)
                return &c;
        return nullptr;
    }

    bool MeshConnectivity::hasCone(int fromDepth, int toDepth) const
    {
        return findCone(fromDepth, toDepth) != nullptr;
    }

    // -----------------------------------------------------------------
    // Support management
    // -----------------------------------------------------------------

    SupportAdj &MeshConnectivity::addSupport(int fromDepth, int toDepth)
    {
        DNDS_assert_info(!hasSupport(fromDepth, toDepth),
                         fmt::format("Support ({}, {}) already exists", fromDepth, toDepth));
        supports.push_back(SupportAdj{fromDepth, toDepth, makeAdjVariant<tAdjPair>()});
        return supports.back();
    }

    SupportAdj *MeshConnectivity::findSupport(int fromDepth, int toDepth)
    {
        for (auto &s : supports)
            if (s.fromDepth == fromDepth && s.toDepth == toDepth)
                return &s;
        return nullptr;
    }

    const SupportAdj *MeshConnectivity::findSupport(int fromDepth, int toDepth) const
    {
        for (const auto &s : supports)
            if (s.fromDepth == fromDepth && s.toDepth == toDepth)
                return &s;
        return nullptr;
    }

    bool MeshConnectivity::hasSupport(int fromDepth, int toDepth) const
    {
        return findSupport(fromDepth, toDepth) != nullptr;
    }

} // namespace DNDS::Geom
