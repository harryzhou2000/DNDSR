#pragma once

// Included after MeshConnectivity query types are declared. Enumeration uses
// callbacks only; both compatibility arrays and native readers share this math.
namespace DNDS::Geom::ConnectivityRows
{
    struct SubEntityRows
    {
        std::vector<std::vector<index>> parent2entity, entity2node, entity2parent;
        std::vector<ElemInfo> entityElemInfo;
        index nEntities = 0;
    };

    template <class ReadNode>
    SubEntityRows EnumerateSubEntities(ReadNode &&readNode, const SubEntityQuery &query,
                                       index nParent, index nNode)
    {
        SubEntityRows result;
        result.parent2entity.resize(size_t(nParent));

        auto &entityElemInfoV = result.entityElemInfo;
        index &nEntities = result.nEntities;
        nEntities = 0;

        // Detached row mathematics; callers choose their output storage.
        auto &ent2nodeVec = result.entity2node;
        auto &ent2parentVec = result.entity2parent;

        // Per-entity creating (parent, sub) pair — needed for matchExtra callback
        struct EntityOrigin
        {
            index parentIdx;
            int subIdx;
        };
        std::vector<EntityOrigin> ent2origin;

        // Reverse index: for each node, which entities contain it.
        // Used for O(1)-amortized deduplication lookup.
        std::vector<std::vector<index>> node2entity(nNode);

        // Scratch buffer for extracted sub-entity nodes
        constexpr int maxSubEntityNodes = 10; // Quad9 face = 9 nodes, padded
        static_assert(maxSubEntityNodes >= 10,
                      "maxSubEntityNodes must be >= 10 (Quad9 face = 9 nodes); "
                      "bump if adding richer elements");
        std::array<index, maxSubEntityNodes> subNodesBuf{};

        const bool hasMatchExtra = bool(query.matchExtra);

        for (index iParent = 0; iParent < nParent; iParent++)
        {
            int nSubs = query.numSubEntities(iParent);
            result.parent2entity[size_t(iParent)].resize(size_t(nSubs));

            // Adapter: wrap parent2node[iParent] row as a function(int) -> index
            auto parentNodeAccessor = [&](int j) -> index
            {
                return readNode(iParent, j);
            };

            for (int iSub = 0; iSub < nSubs; iSub++)
            {
                SubEntityDesc desc = query.describe(iParent, iSub);
                DNDS_assert_info(desc.nNodes <= maxSubEntityNodes,
                                 fmt::format("Interpolate: sub-entity has {} nodes, max is {}",
                                             desc.nNodes, maxSubEntityNodes));

                // Extract sub-entity nodes from parent's node list
                query.extractNodes(iParent, iSub, parentNodeAccessor, subNodesBuf.data());

                // Sort vertices for comparison (first nVertices entries)
                std::vector<index> subVerts(subNodesBuf.begin(),
                                            subNodesBuf.begin() + desc.nVertices);
                std::sort(subVerts.begin(), subVerts.end());

                // Deduplicate: search existing entities via reverse index
                index iFound = -1;
                for (auto iV : subVerts)
                {
                    if (iFound >= 0)
                        break;
                    DNDS_assert_info(iV >= 0 && iV < nNode,
                                     fmt::format("Interpolate: node index {} out of range [0, {})",
                                                 iV, nNode));
                    for (auto iEnt : node2entity[iV])
                    {
                        // Type-tag must match
                        if (entityElemInfoV[iEnt].type != desc.typeTag)
                            continue;

                        // Compare sorted vertices
                        auto &entNodes = ent2nodeVec[iEnt];
                        int nEntVerts = desc.nVertices;

                        std::vector<index> entVerts(entNodes.begin(),
                                                    entNodes.begin() + nEntVerts);
                        std::sort(entVerts.begin(), entVerts.end());

                        if (!std::equal(subVerts.begin(), subVerts.end(),
                                        entVerts.begin(), entVerts.end()))
                            continue;

                        // Vertex match found. Apply optional extra predicate.
                        if (hasMatchExtra)
                        {
                            auto &orig = ent2origin[iEnt];
                            if (!query.matchExtra(iParent, iSub,
                                                  iEnt,
                                                  orig.parentIdx, orig.subIdx))
                                continue; // Rejected by extra predicate
                        }

                        iFound = iEnt;
                        break;
                    }
                }

                if (iFound < 0)
                {
                    // New sub-entity
                    std::vector<index> subNodesVec(subNodesBuf.begin(),
                                                   subNodesBuf.begin() + desc.nNodes);
                    ent2nodeVec.emplace_back(std::move(subNodesVec));
                    ent2parentVec.push_back({iParent});
                    entityElemInfoV.emplace_back(ElemInfo{desc.typeTag, 0});
                    ent2origin.push_back({iParent, iSub});

                    // Register in reverse index (vertices only)
                    for (auto iV : subVerts)
                        node2entity[iV].push_back(nEntities);

                    result.parent2entity[size_t(iParent)][size_t(iSub)] = nEntities;
                    nEntities++;
                }
                else
                {
                    // Existing sub-entity: append parent to the parent list.
                    ent2parentVec[iFound].push_back(iParent);

                    result.parent2entity[size_t(iParent)][size_t(iSub)] = iFound;
                }
            }
        }

        return result;
    }
}
