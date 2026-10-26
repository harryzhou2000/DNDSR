#pragma once

#include "RelationAccess.hpp"
#include "AdjIndexInfo.hpp"
#include "DNDS/Runtime/LegacyArrayBridge.hpp"

namespace DNDS::Geom
{
    template <class Pair>
    RelationReadAccess::RelationReadAccess(AdjKind kind, const AdjPairTracked<Pair> &pair, EntitySpace source, EntitySpace target)
        : _source(std::move(source)), _target(std::move(target)),
          _sourceLease(_source._versions.Read()), _targetLease(_target._versions.Read()),
          _kind(kind), _state(pair.idx.state())
    {
        DNDS_check_throw_info(pair.father && pair.son, "legacy relation arrays are missing");
        DNDS_check_throw_info(pair.father->pLGlobalMapping == _source._global &&
                                  (pair.son->Size() == 0 || pair.trans.pLGhostMapping == _source._ghostMap),
                              "legacy relation row mapping differs from its source space");
        DNDS_check_throw_info(_state != Adj_PointToLocal || pair.idx.mapping() == _target._ghostMap,
                              "legacy relation target mapping differs from its target space");
        _view.owned = Capture(Runtime::ShareLegacyHost(*pair.father), _owned);
        _view.ghost = Capture(Runtime::ShareLegacyHost(*pair.son), _ghost);
        Validate();
    }
    template <rowsize target_rs, rowsize source_rs = NonUniformSize>
    static void narrowAdjToFixed(
        const ArrayAdjacencyPair<source_rs> &source,
        ArrayAdjacencyPair<target_rs> &target,
        index nRows,
        index fill = UnInitIndex)
    {
        static_assert(target_rs > 0, "narrowAdjToFixed: target must be fixed-width");
        for (index i = 0; i < nRows; i++)
        {
            auto row = source.father->operator[](i);
            for (rowsize j = 0; j < target_rs; j++)
                target.father->operator()(i, j) =
                    (j < static_cast<rowsize>(row.size())) ? row[j] : fill;
        }
    }

}
