#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace DNDS::Geom
{
    // =================================================================
    // EntityKind: logical entity roles
    // =================================================================

    /// Logical entity roles in the mesh. Topological depth depends on the
    /// mesh dimension (2D or 3D).
    ///
    /// In 3D: Cell=3, Face=2, Edge=1, Node=0.
    /// In 2D: Cell=2, Face=1, Edge=1 (==Face), Node=0.
    /// Bnd shares Face's depth but is stored separately (zone-labeled subset).
    enum class EntityKind : int8_t
    {
        Cell = 0,
        Face = 1,
        Edge = 2,
        Node = 3,
        Bnd = 4,
        NUM_KINDS = 5,
    };

    /// Resolve EntityKind to topological depth for a given mesh dimension.
    /// In 2D, Edge and Face both resolve to dim-1 = 1.
    /// Bnd resolves to dim-1 (same depth as Face).
    inline int entityDepth(EntityKind kind, int dim)
    {
        switch (kind)
        {
        case EntityKind::Cell:
            return dim;
        case EntityKind::Face:
            return dim - 1;
        case EntityKind::Edge:
            // NOLINTNEXTLINE(bugprone-branch-clone) — Edge depth is always 1 in both 2D and 3D
            return (dim == 2) ? 1 : 1;
        case EntityKind::Node:
            return 0;
        case EntityKind::Bnd:
            return dim - 1; // same depth as Face
        default:
            return -1;
        }
    }

    /// String name for an EntityKind (for diagnostics).
    inline const char *entityKindName(EntityKind kind)
    {
        switch (kind)
        {
        case EntityKind::Cell:
            return "Cell";
        case EntityKind::Face:
            return "Face";
        case EntityKind::Edge:
            return "Edge";
        case EntityKind::Node:
            return "Node";
        case EntityKind::Bnd:
            return "Bnd";
        default:
            return "Unknown";
        }
    }

    // =================================================================
    // AdjacencyDescriptor: named adjacency hop
    // =================================================================

    enum class RelationSelection : uint8_t
    {
        Legacy,
        Incidence,
        Any,
        All,
        Exact,
        Custom
    };
    enum class RelationOrder : uint8_t
    {
        Legacy,
        Topological,
        SortedUnique
    };
    enum class RelationPeriodicPolicy : uint8_t
    {
        Legacy,
        EntityIdentity,
        FrameMatched
    };

    /// Comparable setup descriptor. Legacy constructors are explicit adapters
    /// for historical mesh slots; native construction retains its full path,
    /// selection and provenance rather than guessing from endpoint kinds.
    struct AdjacencyDescriptor
    {
        EntityKind from{EntityKind::Cell};
        EntityKind to{EntityKind::Node};
        EntityKind via{EntityKind::Node};
        std::vector<EntityKind> path;
        RelationSelection selection{RelationSelection::Legacy};
        bool removeSelf{false};
        RelationOrder order{RelationOrder::Legacy};
        RelationPeriodicPolicy periodic{RelationPeriodicPolicy::Legacy};
        std::string semanticName;
        uint32_t semanticVersion{0};
        /// Construction records input definitions separately from row semantics.
        std::string provenance;

        AdjacencyDescriptor() = default;
        /// Historical direct slot adapter. New composed relations must use
        /// Composition(), including when their endpoint kinds differ.
        AdjacencyDescriptor(EntityKind from_, EntityKind to_) : from(from_), to(to_) {}
        /// Historical same-level adapter. For unequal endpoints `via` remains
        /// ignored, preserving old keys; use Composition() for native paths.
        AdjacencyDescriptor(EntityKind from_, EntityKind to_, EntityKind via_) : from(from_), to(to_), via(via_) {}
        [[nodiscard]] bool isIntraLevel() const { return from == to; }
        [[nodiscard]] bool isDirect() const { return from != to; }
        [[nodiscard]] bool IsLegacy() const { return selection == RelationSelection::Legacy; }
        [[nodiscard]] std::vector<EntityKind> Path() const;
        /// Length-delimited canonical wire representation, independent of hashes
        /// and process addresses. Used to compare collective setup schedules.
        [[nodiscard]] std::string Signature() const;
        /// Stable compact descriptor ID for inventories; use full descriptor
        /// equality as well when identity must be collision-free.
        [[nodiscard]] uint64_t DescriptorId() const;
        void Validate() const;
        static AdjacencyDescriptor Composition(const AdjacencyDescriptor &ab, const AdjacencyDescriptor &bc,
                                               RelationSelection selection = RelationSelection::Any, bool removeSelf = false,
                                               std::string name = {}, uint32_t version = 0,
                                               RelationPeriodicPolicy periodic = RelationPeriodicPolicy::EntityIdentity);
        /// Native mathematical inverse with sorted unique output and complete
        /// input provenance; legacy physical-slot adapters remain explicit.
        [[nodiscard]] AdjacencyDescriptor Inverted() const;
        bool operator==(const AdjacencyDescriptor &other) const { return Signature() == other.Signature(); }
        bool operator!=(const AdjacencyDescriptor &other) const { return !(*this == other); }
        bool operator<(const AdjacencyDescriptor &other) const { return Signature() < other.Signature(); }
    };

    struct AdjacencyDescriptorHash
    {
        std::size_t operator()(const AdjacencyDescriptor &kind) const
        {
            return std::size_t(kind.DescriptorId());
        }
    };

    using AdjKind = AdjacencyDescriptor;         // compatibility name
    using AdjKindHash = AdjacencyDescriptorHash; // compatibility name

    /// Convenience constants for common adjacency kinds.
    namespace Adj
    {
        // Direct cones (downward)
        inline const AdjacencyDescriptor Cell2Node{EntityKind::Cell, EntityKind::Node};
        inline const AdjacencyDescriptor Cell2Face{EntityKind::Cell, EntityKind::Face};
        inline const AdjacencyDescriptor Cell2Edge{EntityKind::Cell, EntityKind::Edge};
        inline const AdjacencyDescriptor Face2Node{EntityKind::Face, EntityKind::Node};
        inline const AdjacencyDescriptor Face2Edge{EntityKind::Face, EntityKind::Edge};
        inline const AdjacencyDescriptor Edge2Node{EntityKind::Edge, EntityKind::Node};
        inline const AdjacencyDescriptor Bnd2Node{EntityKind::Bnd, EntityKind::Node};

        // Direct supports (upward)
        inline const AdjacencyDescriptor Node2Cell{EntityKind::Node, EntityKind::Cell};
        inline const AdjacencyDescriptor Node2Face{EntityKind::Node, EntityKind::Face};
        inline const AdjacencyDescriptor Node2Edge{EntityKind::Node, EntityKind::Edge};
        inline const AdjacencyDescriptor Node2Bnd{EntityKind::Node, EntityKind::Bnd};
        inline const AdjacencyDescriptor Face2Cell{EntityKind::Face, EntityKind::Cell};
        inline const AdjacencyDescriptor Edge2Face{EntityKind::Edge, EntityKind::Face};
        inline const AdjacencyDescriptor Edge2Cell{EntityKind::Edge, EntityKind::Cell};
        inline const AdjacencyDescriptor Bnd2Cell{EntityKind::Bnd, EntityKind::Cell};
        /// Ordered physical own/partner slots, distinct from the legacy
        /// Bnd2Cell facade's periodic own/self convention.
        inline const AdjacencyDescriptor Bnd2CellCoupled = []
        {
            auto key = AdjacencyDescriptor::Composition(Bnd2Node, Cell2Node.Inverted(), RelationSelection::Custom, false,
                                                        "geom.boundary.coupled-physical-slots", 1, RelationPeriodicPolicy::FrameMatched);
            key.order = RelationOrder::Topological;
            return key;
        }();
        inline const AdjacencyDescriptor Bnd2Face{EntityKind::Bnd, EntityKind::Face};
        inline const AdjacencyDescriptor Face2Bnd{EntityKind::Face, EntityKind::Bnd};

        // Intra-level (composed), default via Node
        inline const AdjacencyDescriptor Cell2Cell{EntityKind::Cell, EntityKind::Cell, EntityKind::Node};
        inline const AdjacencyDescriptor Bnd2Bnd{EntityKind::Bnd, EntityKind::Bnd, EntityKind::Node};
        inline const AdjacencyDescriptor Face2Face{EntityKind::Face, EntityKind::Face, EntityKind::Node};

        // Intra-level with explicit intermediary
        inline const AdjacencyDescriptor Cell2CellFace{EntityKind::Cell, EntityKind::Cell, EntityKind::Face};
    } // namespace Adj

    /// Format an AdjacencyDescriptor as a diagnostic string, e.g. "Cell2Node", "Cell2Cell(Node)".
    std::string adjKindName(const AdjacencyDescriptor &kind);

}
