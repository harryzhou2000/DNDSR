#pragma once

#include "Geom/Access/MeshView.hpp"
#include "Geom/Connectivity/MeshTopologyTypes.hpp"
#include <array>

namespace DNDS::Geom::MeshTransitionDetail
{
    /// Private canonical facade bindings in MeshRelation ordinal order.
    /// Assembly and transitions must use exactly the same descriptor identity;
    /// changing the semantic name/version is a publication compatibility change.
    inline const std::array<AdjacencyDescriptor, size_t(MeshRelation::Count)> &Bindings()
    {
        static const std::array<AdjacencyDescriptor, size_t(MeshRelation::Count)> result{
            Adj::Cell2Node, Adj::Bnd2Node, Adj::Bnd2Cell, Adj::Cell2Cell,
            Adj::Node2Cell, Adj::Node2Bnd, Adj::Cell2Face, Adj::Face2Node,
            Adj::Face2Cell, Adj::Face2Bnd, Adj::Bnd2Face, Adj::Cell2Edge,
            Adj::Edge2Node, Adj::Edge2Cell, Adj::Cell2CellFace};
        return result;
    }

    inline AdjacencyDescriptor FacadeKey(size_t slot)
    {
        auto key = Bindings()[slot];
        key.path = key.Path();
        key.selection = RelationSelection::Incidence;
        key.order = RelationOrder::Topological;
        key.periodic = RelationPeriodicPolicy::EntityIdentity;
        key.semanticName = "geom.transition.facade." + std::to_string(slot);
        key.semanticVersion = 1;
        return key;
    }
}
