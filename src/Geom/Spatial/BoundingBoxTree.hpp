#pragma once

#include "Geom/Geometric.hpp"
#include <memory>

namespace DNDS::Geom
{
    struct SpatialBox
    {
        tPoint lower = tPoint::Constant(std::numeric_limits<real>::infinity()),
               upper = tPoint::Constant(-std::numeric_limits<real>::infinity());
        void Add(const tPoint &point)
        {
            lower = lower.cwiseMin(point);
            upper = upper.cwiseMax(point);
        }
        void Add(const SpatialBox &box)
        {
            Add(box.lower);
            Add(box.upper);
        }
        real SquaredDistance(const tPoint &point) const
        {
            return (lower - point).cwiseMax(0.).squaredNorm() + (point - upper).cwiseMax(0.).squaredNorm();
        }
        bool Contains(const tPoint &point, real tolerance = 0) const
        {
            return (point.array() >= lower.array() - tolerance).all() && (point.array() <= upper.array() + tolerance).all();
        }
        bool Intersects(const SpatialBox &other, real tolerance = 0) const
        {
            return (lower.array() <= other.upper.array() + tolerance).all() && (other.lower.array() <= upper.array() + tolerance).all();
        }
    };
    enum class BoundingBoxBackend
    {
        Native,
        CGAL
    };
    /// Immutable CPU candidate index, independent of mesh entity types. Both
    /// backends own their boxes and emit input ordinals with the same inclusive
    /// squared point/box-gap predicate. Traversal order is unspecified. Exact
    /// element projection/containment belongs to the caller, not this index.
    /// Copies share a fully built read-only index and permit concurrent queries.
    /// CGAL implementation/types remain private; no device execution is implied.
    class BoundingBoxTree
    {
        struct Data;
        std::shared_ptr<const Data> _data;
        using Visitor = void (*)(void *, index);
        void QueryImpl(const tPoint &point, real gap, void *context, Visitor visit) const;

    public:
        explicit BoundingBoxTree(const std::vector<SpatialBox> &boxes, BoundingBoxBackend backend = BoundingBoxBackend::CGAL);
        index Size() const;
        BoundingBoxBackend Backend() const;
        /// Synchronous visit, no output allocation and no retained callback.
        /// Query inputs must be finite with gap >= 0; invalid boxes/backends
        /// fail during construction. Point/line/planar boxes are supported.
        /// Exceptions from visit propagate; no shared tree state is modified.
        /// The operation pins its index through all callbacks. Replacing a
        /// handle on another thread still requires ordinary synchronization.
        template <class F>
        void Query(const tPoint &point, real gap, F &&visit) const
        {
            auto callback = [&](index item)
            { visit(item); };
            QueryImpl(point, gap, &callback, [](void *context, index item)
                      { (*static_cast<decltype(callback) *>(context))(item); });
        }
    };
}
