#include "Geom/Connectivity/MeshConnectivity.hpp"
#include "Geom/Connectivity/AdjacencyRegistry.hpp"

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

    std::vector<EntityKind> AdjacencyDescriptor::Path() const
    {
        if (!path.empty())
            return path;
        return isIntraLevel() ? std::vector<EntityKind>{from, via, to} : std::vector<EntityKind>{from, to};
    }

    std::string AdjacencyDescriptor::Signature() const
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

    uint64_t AdjacencyDescriptor::DescriptorId() const
    {
        uint64_t value = 14695981039346656037ULL;
        for (unsigned char byte : Signature())
        {
            value ^= byte;
            value *= 1099511628211ULL;
        }
        return value;
    }

    void AdjacencyDescriptor::Validate() const
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

    AdjacencyDescriptor AdjacencyDescriptor::Composition(const AdjacencyDescriptor &ab, const AdjacencyDescriptor &bc, RelationSelection selection_,
                                                         bool removeSelf_, std::string name, uint32_t version, RelationPeriodicPolicy periodic_)
    {
        ab.Validate();
        bc.Validate();
        DNDS_check_throw_info(ab.to == bc.from, "relation descriptor composition endpoint mismatch");
        AdjacencyDescriptor result(ab.from, bc.to, ab.to);
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

    AdjacencyDescriptor AdjacencyDescriptor::Inverted() const
    {
        Validate();
        // Mathematical inversion produces sorted set-valued rows. It never
        // claims an ordered legacy physical slot, even for direct cones.
        // The complete input definition remains explicit construction provenance.
        AdjacencyDescriptor result = *this;
        result.from = to;
        result.to = from;
        result.path = Path();
        std::reverse(result.path.begin(), result.path.end());
        if (IsLegacy())
        {
            result.selection = isIntraLevel() ? RelationSelection::Any : RelationSelection::Incidence;
            result.periodic = RelationPeriodicPolicy::EntityIdentity;
            result.semanticName = "geom.inverse";
            result.semanticVersion = 1;
        }
        result.order = RelationOrder::SortedUnique;
        result.provenance = "inverse:" + Signature();
        result.Validate();
        return result;
    }

    std::string adjKindName(const AdjacencyDescriptor &kind)
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
}

namespace DNDS::Geom
{
    MeshAdjacency MeshConnectivity::Inverse(const MeshAdjacencyRead &cone, const MPIInfo &mpi)
    {
        return AdjacencyRegistry::Inverse(cone, mpi);
    }

    MeshAdjacency MeshConnectivity::Compose(const MeshAdjacencyRead &AB, const MeshAdjacencyRead &BC, bool removeSelf)
    {
        return AdjacencyRegistry::Compose(AB, BC, removeSelf);
    }

    MeshAdjacency MeshConnectivity::ComposeAll(const MeshAdjacencyRead &AB, const MeshAdjacencyRead &BC, bool removeSelf)
    {
        return AdjacencyRegistry::ComposeAll(AB, BC, removeSelf);
    }

    MeshAdjacency MeshConnectivity::ComposeFiltered(const MeshAdjacencyRead &AB, const MeshAdjacencyRead &BC,
                                                    const AdjacencyDescriptor &resultKind,
                                                    const std::function<bool(index, index, int)> &predicate,
                                                    const std::function<bool(index, index, const std::vector<index> &)> &matchExtra)
    {
        return AdjacencyRegistry::ComposeFiltered(AB, BC, resultKind, predicate, matchExtra);
    }

}
