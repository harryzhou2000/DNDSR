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

        GhostSpec result;
        // Collection is endpoint-only. Emit one explicit chain per requested
        // depth to retain cumulative ring semantics in the default pipeline.
        for (int depth = 1; depth <= nLayers; ++depth)
        {
            std::vector<AdjacencyDescriptor> cellHops(size_t(depth), Cell2Cell);
            result.chains.push_back({EntityKind::Cell, cellHops, EntityKind::Cell});
            cellHops.push_back(Cell2Node);
            result.chains.push_back({EntityKind::Cell, std::move(cellHops), EntityKind::Node});
        }
        result.chains.push_back({EntityKind::Bnd, {Bnd2Node}, EntityKind::Node});
        result.chains.push_back({EntityKind::Bnd, {Bnd2Node, Node2Bnd}, EntityKind::Bnd});
        result.chains.push_back({EntityKind::Bnd, {Bnd2Node, Node2Bnd, Bnd2Node}, EntityKind::Node});
        return result;
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
            const AdjacencyDescriptor &hop = chain.hops[hi];
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

            // Only the exact endpoint of this input chain collects. A node can
            // be an endpoint for one chain and an intermediate node for another;
            // the aggregate marker is therefore independent of leaf-ness.
            if (hi + 1 == chain.hops.size())
                child->collect = true;

            current = child;
        }
    }

    CompiledGhostTree CompiledGhostTree::compile(const GhostSpec &spec)
    {
        Data tree;
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

            DNDS_check_throw_info(nextId < INT_MAX, "compiled ghost program exceeds node ID range");
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

        // Length-prefix complete descriptor signatures so diagnostic names and
        // user-defined descriptors cannot make different programs agree.
        std::ostringstream signature;
        signature << "ghost-program-v1:" << tree.totalNodes << ':' << tree.maxLevel << ':';
        for (const auto &level : tree.levels)
        {
            signature << level.size() << ':';
            for (const auto &entry : level)
            {
                const auto hop = entry.parentId < 0 ? std::string{} : entry.hop.Signature();
                signature << entry.nodeId << ':' << entry.parentId << ':' << int(entry.kind) << ':'
                          << entry.collect << ':' << entry.hasChildren << ':' << hop.size() << ':' << hop;
            }
        }
        tree.signature = signature.str();
        return CompiledGhostTree(std::move(tree));
    }

    // =================================================================
    // CompiledGhostTree helper methods
    // =================================================================

    static void collectRequiredAdjsHelper(
        const GhostTreeNode &node,
        std::unordered_set<AdjacencyDescriptor, AdjacencyDescriptorHash> &out)
    {
        for (const auto &child : node.children)
        {
            out.insert(child.hop);
            collectRequiredAdjsHelper(child, out);
        }
    }

    std::unordered_set<AdjacencyDescriptor, AdjacencyDescriptorHash> CompiledGhostTree::requiredAdjs() const
    {
        std::unordered_set<AdjacencyDescriptor, AdjacencyDescriptorHash> result;
        for (const auto &root : Roots())
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
        for (const auto &root : Roots())
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
        for (const auto &root : Roots())
            dumpNode(root, oss, 0);
        return oss.str();
    }

    // =================================================================
    // GhostResult helpers
    // =================================================================

    // (hasGhosts and totalGhosts are inline in the header)

}
