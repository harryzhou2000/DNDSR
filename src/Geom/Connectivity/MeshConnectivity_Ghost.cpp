/// @file MeshConnectivity_Ghost.cpp
/// @brief Ghost chain compilation and BFS evaluation.

#include "Geom/Connectivity/MeshConnectivity.hpp"

#include <algorithm>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <fmt/core.h>

namespace DNDS::Geom
{
    // =================================================================
    // GhostSpec::defaultPrimary
    // =================================================================

    GhostSpec GhostSpec::defaultPrimary(int nLayers)
    {
        using namespace Adj;
        DNDS_assert(nLayers >= 1);

        // Cell chain: nLayers hops of Cell2Cell
        std::vector<AdjKind> cellHops(nLayers, Cell2Cell);

        // Node chain: nLayers hops of Cell2Cell + Cell2Node
        std::vector<AdjKind> nodeHops(nLayers, Cell2Cell);
        nodeHops.push_back(Cell2Node);

        return GhostSpec{{
            {EntityKind::Cell, cellHops, EntityKind::Cell},
            {EntityKind::Cell, nodeHops, EntityKind::Node},
            {EntityKind::Bnd, {Bnd2Node, Node2Bnd}, EntityKind::Bnd},
            {EntityKind::Bnd, {Bnd2Node, Node2Bnd, Bnd2Node}, EntityKind::Node},
        }};
    }

    // =================================================================
    // CompiledGhostTree::compile
    // =================================================================

    /// Insert a chain into a trie rooted at `roots`. Merges common prefixes.
    static void insertChainIntoTrie(
        std::vector<GhostTreeNode> &roots,
        const GhostChain &chain)
    {
        // Validate chain consistency and complete semantic descriptors.
        for (const auto &hop : chain.hops)
            hop.Validate();
        if (chain.hops.empty())
            throw std::runtime_error("GhostChain: empty hop list");
        if (chain.anchor != chain.hops.front().from)
            throw std::runtime_error(fmt::format(
                "GhostChain: anchor {} != first hop from {}",
                entityKindName(chain.anchor),
                entityKindName(chain.hops.front().from)));
        if (chain.target != chain.hops.back().to)
            throw std::runtime_error(fmt::format(
                "GhostChain: target {} != last hop to {}",
                entityKindName(chain.target),
                entityKindName(chain.hops.back().to)));
        for (size_t i = 0; i + 1 < chain.hops.size(); i++)
        {
            if (chain.hops[i].to != chain.hops[i + 1].from)
                throw std::runtime_error(fmt::format(
                    "GhostChain: hop[{}].to ({}) != hop[{}].from ({})",
                    i, entityKindName(chain.hops[i].to),
                    i + 1, entityKindName(chain.hops[i + 1].from)));
        }

        // Find or create the root node for this chain's anchor.
        GhostTreeNode *current = nullptr;
        for (auto &root : roots)
        {
            if (root.kind == chain.anchor)
            {
                current = &root;
                break;
            }
        }
        if (!current)
        {
            roots.push_back(GhostTreeNode{chain.anchor, {}, false, 0, {}});
            current = &roots.back();
        }

        // Walk down the trie, creating nodes as needed.
        for (size_t hi = 0; hi < chain.hops.size(); hi++)
        {
            const AdjKind &hop = chain.hops[hi];
            int childLevel = current->level + 1;
            EntityKind childKind = hop.to;

            // Look for existing child with same hop.
            GhostTreeNode *child = nullptr;
            for (auto &c : current->children)
            {
                if (c.hop == hop && c.kind == childKind)
                {
                    child = &c;
                    break;
                }
            }
            if (!child)
            {
                current->children.push_back(
                    GhostTreeNode{childKind, hop, false, childLevel, {}});
                child = &current->children.back();
            }

            // If this is the last hop, mark as collect.
            // Also mark intermediate hops whose entity kind matches the
            // chain target -- this enables cumulative ghost collection for
            // multi-hop same-kind chains (e.g., Cell2Cell -> Cell2Cell).
            if (hi + 1 == chain.hops.size() || childKind == chain.target)
                child->collect = true;

            current = child;
        }
    }

    CompiledGhostTree CompiledGhostTree::compile(const GhostSpec &spec)
    {
        CompiledGhostTree tree;
        for (const auto &chain : spec.chains)
            insertChainIntoTrie(tree.roots, chain);

        // Assign IDs and parent IDs, compute maxLevel, build per-level lists.
        int nextId = 0;
        tree.maxLevel = 0;

        // BFS to assign IDs and build levels.
        // Use a queue of (node pointer, parent ID).
        struct QEntry
        {
            GhostTreeNode *node;
            int parentId;
        };
        std::queue<QEntry> q;
        for (auto &root : tree.roots)
        {
            root.parentId = -1;
            q.push({&root, -1});
        }

        while (!q.empty())
        {
            auto [node, pid] = q.front();
            q.pop();

            node->id = nextId++;
            node->parentId = pid;

            if (node->level > tree.maxLevel)
                tree.maxLevel = node->level;

            for (auto &child : node->children)
                q.push({&child, node->id});
        }

        tree.totalNodes = nextId;

        // Build per-level lists.
        tree.levels.resize(tree.maxLevel + 1);
        std::function<void(const GhostTreeNode &)> buildLevels =
            [&](const GhostTreeNode &n)
        {
            tree.levels[n.level].push_back(LevelEntry{
                n.id,
                n.parentId,
                n.kind,
                n.hop,
                n.collect,
                !n.children.empty(),
            });
            for (const auto &child : n.children)
                buildLevels(child);
        };
        for (const auto &root : tree.roots)
            buildLevels(root);

        return tree;
    }

    // =================================================================
    // CompiledGhostTree helper methods
    // =================================================================

    static void collectRequiredAdjsHelper(
        const GhostTreeNode &node,
        std::unordered_set<AdjKind, AdjKindHash> &out)
    {
        for (const auto &child : node.children)
        {
            out.insert(child.hop);
            collectRequiredAdjsHelper(child, out);
        }
    }

    std::unordered_set<AdjKind, AdjKindHash> CompiledGhostTree::requiredAdjs() const
    {
        std::unordered_set<AdjKind, AdjKindHash> result;
        for (const auto &root : roots)
            collectRequiredAdjsHelper(root, result);
        return result;
    }

    static void collectCollectedKindsHelper(
        const GhostTreeNode &node,
        std::unordered_set<EntityKind> &out)
    {
        if (node.collect)
            out.insert(node.kind);
        for (const auto &child : node.children)
            collectCollectedKindsHelper(child, out);
    }

    std::unordered_set<EntityKind> CompiledGhostTree::collectedKinds() const
    {
        std::unordered_set<EntityKind> result;
        for (const auto &root : roots)
            collectCollectedKindsHelper(root, result);
        return result;
    }

    // =================================================================
    // CompiledGhostTree::dump
    // =================================================================

    static void dumpNode(
        const GhostTreeNode &node,
        std::ostringstream &oss,
        int indent)
    {
        for (int i = 0; i < indent; i++)
            oss << "  ";
        if (indent > 0)
            oss << "\\-[" << adjKindName(node.hop) << "]-> ";
        oss << entityKindName(node.kind)
            << " (level=" << node.level;
        if (node.collect)
            oss << ", COLLECT";
        oss << ")\n";
        for (const auto &child : node.children)
            dumpNode(child, oss, indent + 1);
    }

    std::string CompiledGhostTree::dump() const
    {
        std::ostringstream oss;
        for (const auto &root : roots)
            dumpNode(root, oss, 0);
        return oss.str();
    }

    // =================================================================
    // GhostResult helpers
    // =================================================================

    // (hasGhosts and totalGhosts are inline in the header)

}
