#pragma once
#include "MeshRelationTypes.hpp"
#include "MeshTopologyTypes.hpp"
#include <functional>
namespace DNDS::Geom
{
    class RelationReadAccess;
    class NativeRelation;
    /// Compatibility spelling for the native setup mathematics. Storage ownership,
    /// relation registration and sparse ghost evaluation belong to NativeConnectivity.
    struct MeshConnectivity
    {
        static NativeRelation Inverse(const RelationReadAccess &, const MPIInfo &);
        static NativeRelation Compose(const RelationReadAccess &, const RelationReadAccess &, bool removeSelf = false);
        static NativeRelation ComposeAll(const RelationReadAccess &, const RelationReadAccess &, bool removeSelf = false);
        static NativeRelation ComposeFiltered(const RelationReadAccess &, const RelationReadAccess &, const AdjKind &,
                                              const std::function<bool(index, index, int)> &,
                                              const std::function<bool(index, index, const std::vector<index> &)> &extra = {});
    };
}
