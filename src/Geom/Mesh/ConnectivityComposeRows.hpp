#pragma once

#include "DNDS/Defines.hpp"
#include <algorithm>
#include <functional>
#include <unordered_map>
#include <vector>

namespace DNDS::Geom::ConnectivityRows
{
    /// Storage-neutral row mathematics shared by legacy and native DSL
    /// adapters. Readers expose entries in global entity spaces; writers
    /// receive each deterministic sorted row exactly once.
    template <class ReadAB, class ReadBC, class ResolveB, class GlobalA, class Predicate, class Write>
    void Compose(index nA, ReadAB &&readAB, ReadBC &&readBC, ResolveB &&resolveB,
                 GlobalA &&globalA, Predicate &&pred, Write &&write,
                 const std::function<bool(index, index, const std::vector<index> &)> &matchExtra = {})
    {
        for (index a = 0; a < nA; ++a)
        {
            const auto aGlobal = globalA(a);
            std::unordered_map<index, int> counts;
            std::unordered_map<index, std::vector<index>> shared;
            for (auto b : readAB(a))
            {
                if (b == UnInitIndex)
                    continue;
                auto localB = resolveB(b);
                for (auto c : readBC(localB))
                {
                    if (c == UnInitIndex)
                        continue;
                    ++counts[c];
                    if (matchExtra)
                        shared[c].push_back(b);
                }
            }
            std::vector<index> accepted;
            for (auto &[c, count] : counts)
                if (pred(aGlobal, c, count) && (!matchExtra || matchExtra(a, c, shared[c])))
                    accepted.push_back(c);
            std::sort(accepted.begin(), accepted.end());
            write(a, accepted);
        }
    }

}
