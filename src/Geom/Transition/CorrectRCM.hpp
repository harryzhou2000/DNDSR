#pragma once

#ifndef CORRECTRCM_H
#    define CORRECTRCM_H

#    include <iostream>
#    include <map>
#    include <exception>
#    include <cstdint>
#    include <vector>
#    include <utility>
#    include <algorithm>
#    include <type_traits>
#    include <optional>

namespace CorrectRCM
{
    struct hash_pair
    {
        template <class T1, class T2>
        size_t operator()(const std::pair<T1, T2> &p) const
        {
            size_t hash1 = std::hash<T1>{}(p.first);
            size_t hash2 = std::hash<T2>{}(p.second);
            return hash1 ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
        }
    };

    template <typename tGraphAdjFunctor>
    class UndirectedGraphProxy
    {
    public:
        using index = int64_t;

    private:
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
        const tGraphAdjFunctor &GraphAdjFunctor;
        index nVertices;

    public:
        UndirectedGraphProxy(const tGraphAdjFunctor &graphAdjFunctor, index nVertices)
            : GraphAdjFunctor(graphAdjFunctor), nVertices(nVertices) {}

        [[nodiscard]] index GetNVertices() const { return nVertices; }

        [[nodiscard]] int CheckAdj(bool checkParallel = true, bool checkSymmetry = true) const
        {
            int invalid = 0;
#    pragma omp parallel for if (nVertices >= 4096) schedule(static) reduction(| : invalid)
            for (index i = 0; i < nVertices; ++i)
            {
                const auto row = GetAdj(i);
                for (auto entry : row)
                {
                    const index j = index(entry);
                    if (j == i)
                        continue;
                    if (checkParallel)
                    {
                        size_t multiplicity = 0;
                        for (auto other : row)
                            multiplicity += index(other) == j;
                        if (multiplicity != 1)
                        {
                            invalid |= 1;
                            continue;
                        }
                    }
                    if (checkSymmetry)
                    {
                        bool reverse = false;
                        for (auto other : GetAdj(j))
                            reverse = reverse || index(other) == i;
                        if (!reverse)
                            invalid |= 2;
                    }
                }
            }
            if (invalid & 1)
                throw std::runtime_error("Graph not having unique directed edges");
            if (invalid & 2)
                throw std::runtime_error("Graph not having symmetric edge pairs");
            return 0;
        }

        [[nodiscard]] auto GetAdj(index i) const { return GraphAdjFunctor(i); }

    private:
        template <typename iter0, typename iter1, typename TFInLayerCompare, typename TLayer>
        std::enable_if_t<
            std::is_same_v<std::remove_reference_t<TFInLayerCompare>, int>> static inlayer_sort(iter0 &&begin, iter1 &&end, TFInLayerCompare &&comp, TLayer &&layer)
        {
        }

        template <typename iter0, typename iter1, typename TFInLayerCompare, typename TLayer>
        std::enable_if_t<
            !std::is_same_v<std::remove_reference_t<TFInLayerCompare>, int>> static inlayer_sort(iter0 &&begin, iter1 &&end, TFInLayerCompare &&comp, TLayer &&layer)
        {
            std::stable_sort(std::forward<iter0>(begin), std::forward<iter1>(end),
                             [&](index i, index j)
                             { return comp(i, j, std::forward<TLayer>(layer)); });
        }

    public:
        template <typename TFNode, typename TFInLayerCompare = int>
        std::vector<index> BreadthFirstSearch(TFNode &&FNode, index root = 0, TFInLayerCompare &&FInLayerCompare = 0) const
        {
            if (root >= nVertices || root < 0)
                throw std::range_error("Invalid root vertex");
            std::vector<index> layer(nVertices, -1);
            layer.at(root) = 0;
            std::vector<index> curLayerQueue;
            std::vector<index> nextLayerQueue;
            curLayerQueue.reserve(32), curLayerQueue.push_back(root);
            nextLayerQueue.reserve(32);
            for (index iLayer = 0; iLayer < nVertices; iLayer++)
            {
                if (curLayerQueue.empty())
                    break;
                for (auto i : curLayerQueue)
                {
                    FNode(i, iLayer);
                    auto nextLayerQueueSiz0 = nextLayerQueue.size();
                    for (auto j : GetAdj(i))
                    {
                        if (layer.at(j) == -1)
                            nextLayerQueue.push_back(j), layer.at(j) = iLayer + 1;
                    }
                    inlayer_sort(nextLayerQueue.begin() + nextLayerQueueSiz0, nextLayerQueue.end(),
                                 std::forward<TFInLayerCompare>(FInLayerCompare), layer);
                }
                std::swap(curLayerQueue, nextLayerQueue);
                nextLayerQueue.clear();
            }
            return layer;
        }
    };

    template <typename TGraph, typename TFFiller, typename TIndex>
    void CuthillMcKeeOrdering(TGraph &&G, TFFiller &&FFiller, TIndex &&root, std::optional<std::reference_wrapper<std::ostream>> os = {})
    {
        if (os.has_value())
            os.value().get() << "CorrectRCM::CuthillMcKeeOrdering: Start" << std::endl;
        using index = typename std::remove_reference_t<TGraph>::index;
        const index vertices = G.GetNVertices();
        if (vertices < 0)
            throw std::range_error("RCM graph has a negative vertex count");
        std::vector<uint8_t> unvisited(size_t(vertices), 1);
        index remaining = vertices;
        index nextUnvisited = 0;

        index cTop{0};
        index nextRoot = root;

        for (index iIter = 0; iIter < vertices; iIter++)
        {
            auto cRootDegree = G.GetAdj(nextRoot).size();
            auto cRootLayer = 0;
            G.BreadthFirstSearch(
                [&](auto i, auto iLayer)
                {
                    if (iLayer > cRootLayer)
                        cRootDegree = G.GetAdj(i).size(), cRootLayer = iLayer, nextRoot = i;
                    if (iLayer == cRootLayer)
                        if (G.GetAdj(i).size() < cRootDegree)
                            cRootDegree = G.GetAdj(i).size(), nextRoot = i;
                },
                std::forward<TIndex>(nextRoot));

            G.BreadthFirstSearch(
                [&](auto i, auto iLayer)
                {
                    FFiller(i) = cTop++;
                    if (i < 0 || i >= vertices || !unvisited[size_t(i)])
                        throw std::logic_error("unvisited set not correct");
                    unvisited[size_t(i)] = 0;
                    --remaining;
                },
                std::forward<TIndex>(nextRoot),
                [&](index i, index j, const std::vector<index> &layer) -> bool
                {
                    index iPredMin = G.GetNVertices();
                    index jPredMin = G.GetNVertices();
                    for (auto in : G.GetAdj(i))
                        if (layer.at(in) >= 0)
                            iPredMin = std::min(iPredMin, index(FFiller(in)));
                    for (auto jn : G.GetAdj(j))
                        if (layer.at(jn) >= 0)
                            jPredMin = std::min(jPredMin, index(FFiller(jn)));
                    return iPredMin == jPredMin ? G.GetAdj(i).size() < G.GetAdj(j).size()
                                                : iPredMin < jPredMin;
                });
            if (os.has_value())
                os.value().get() << "  Part [" << iIter << "], " << "processed [" << cTop << "/" << G.GetNVertices() << "]" << std::endl;
            if (!remaining)
                break;
            while (nextUnvisited < vertices && !unvisited[size_t(nextUnvisited)])
                ++nextUnvisited;
            if (nextUnvisited >= vertices)
                throw std::logic_error("unvisited cursor incomplete");
            nextRoot = nextUnvisited; // we handle disjoint graphs
        }
        if (os.has_value())
            os.value().get() << "CorrectRCM::CuthillMcKeeOrdering: Finished" << std::endl;
    }

} // namespace CorrectRCM

#endif
