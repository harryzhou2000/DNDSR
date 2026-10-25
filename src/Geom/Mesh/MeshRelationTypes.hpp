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
    // AdjKind: named adjacency hop
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
    struct AdjKind
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

        AdjKind() = default;
        /// Historical direct slot adapter. New composed relations must use
        /// Composition(), including when their endpoint kinds differ.
        AdjKind(EntityKind from_, EntityKind to_) : from(from_), to(to_) {}
        /// Historical same-level adapter. For unequal endpoints `via` remains
        /// ignored, preserving old keys; use Composition() for native paths.
        AdjKind(EntityKind from_, EntityKind to_, EntityKind via_) : from(from_), to(to_), via(via_) {}
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
        static AdjKind Composition(const AdjKind &ab, const AdjKind &bc,
                                   RelationSelection selection = RelationSelection::Any, bool removeSelf = false,
                                   std::string name = {}, uint32_t version = 0,
                                   RelationPeriodicPolicy periodic = RelationPeriodicPolicy::EntityIdentity);
        /// Native mathematical inverse with sorted unique output and complete
        /// input provenance; legacy physical-slot adapters remain explicit.
        [[nodiscard]] AdjKind Inverted() const;
        bool operator==(const AdjKind &other) const { return Signature() == other.Signature(); }
        bool operator!=(const AdjKind &other) const { return !(*this == other); }
        bool operator<(const AdjKind &other) const { return Signature() < other.Signature(); }
    };

    struct AdjKindHash
    {
        std::size_t operator()(const AdjKind &kind) const
        {
            return std::size_t(kind.DescriptorId());
        }
    };

    /// Convenience constants for common adjacency kinds.
    namespace Adj
    {
        // Direct cones (downward)
        inline const AdjKind Cell2Node{EntityKind::Cell, EntityKind::Node};
        inline const AdjKind Cell2Face{EntityKind::Cell, EntityKind::Face};
        inline const AdjKind Cell2Edge{EntityKind::Cell, EntityKind::Edge};
        inline const AdjKind Face2Node{EntityKind::Face, EntityKind::Node};
        inline const AdjKind Face2Edge{EntityKind::Face, EntityKind::Edge};
        inline const AdjKind Edge2Node{EntityKind::Edge, EntityKind::Node};
        inline const AdjKind Bnd2Node{EntityKind::Bnd, EntityKind::Node};

        // Direct supports (upward)
        inline const AdjKind Node2Cell{EntityKind::Node, EntityKind::Cell};
        inline const AdjKind Node2Face{EntityKind::Node, EntityKind::Face};
        inline const AdjKind Node2Edge{EntityKind::Node, EntityKind::Edge};
        inline const AdjKind Node2Bnd{EntityKind::Node, EntityKind::Bnd};
        inline const AdjKind Face2Cell{EntityKind::Face, EntityKind::Cell};
        inline const AdjKind Edge2Face{EntityKind::Edge, EntityKind::Face};
        inline const AdjKind Edge2Cell{EntityKind::Edge, EntityKind::Cell};
        inline const AdjKind Bnd2Cell{EntityKind::Bnd, EntityKind::Cell};
        inline const AdjKind Bnd2Face{EntityKind::Bnd, EntityKind::Face};
        inline const AdjKind Face2Bnd{EntityKind::Face, EntityKind::Bnd};

        // Intra-level (composed), default via Node
        inline const AdjKind Cell2Cell{EntityKind::Cell, EntityKind::Cell, EntityKind::Node};
        inline const AdjKind Bnd2Bnd{EntityKind::Bnd, EntityKind::Bnd, EntityKind::Node};
        inline const AdjKind Face2Face{EntityKind::Face, EntityKind::Face, EntityKind::Node};

        // Intra-level with explicit intermediary
        inline const AdjKind Cell2CellFace{EntityKind::Cell, EntityKind::Cell, EntityKind::Face};
    } // namespace Adj

    /// Format an AdjKind as a diagnostic string, e.g. "Cell2Node", "Cell2Cell(Node)".
    std::string adjKindName(const AdjKind &kind);

}
