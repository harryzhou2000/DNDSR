#include "BoundingBoxTree.hpp"
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits_3.h>
#include <CGAL/Simple_cartesian.h>
#include <numeric>

namespace DNDS::Geom
{
    namespace
    {
        using Kernel = CGAL::Simple_cartesian<real>;
        using BoxIterator = std::vector<SpatialBox>::const_iterator;
        struct BoxPrimitive
        {
            using Id = BoxIterator;
            using Datum = Kernel::Iso_cuboid_3;
            using Point = Kernel::Point_3;
            BoxIterator source;
            explicit BoxPrimitive(BoxIterator iterator) : source(iterator) {}
            Id id() const { return source; }
            Datum datum() const
            {
                return {{source->lower[0], source->lower[1], source->lower[2]},
                        {source->upper[0], source->upper[1], source->upper[2]}};
            }
            Point reference_point() const
            {
                const tPoint center = 0.5 * source->lower + 0.5 * source->upper;
                return {center[0], center[1], center[2]};
            }
        };
        struct CentroidSplit
        {
            template <class Iterator>
            void operator()(Iterator first, Iterator last, const CGAL::Bbox_3 &) const
            {
                // Primitive extents can be long while centers coincide along
                // that axis. Use center spread and stable input-ordinal ties.
                SpatialBox centers;
                for (auto iterator = first; iterator != last; ++iterator)
                    centers.Add(0.5 * iterator->source->lower + 0.5 * iterator->source->upper);
                int axis = 0;
                (centers.upper - centers.lower).maxCoeff(&axis);
                std::nth_element(first, first + (last - first) / 2, last, [axis](const auto &a, const auto &b)
                                 {
                    const real ca = 0.5 * a.source->lower[axis] + 0.5 * a.source->upper[axis];
                    const real cb = 0.5 * b.source->lower[axis] + 0.5 * b.source->upper[axis];
                    return ca == cb ? a.source < b.source : ca < cb; });
            }
        };
        struct BoxTraits : CGAL::AABB_traits_3<Kernel, BoxPrimitive>
        {
            using Split_primitives = CentroidSplit;
            Split_primitives split_primitives_object() const { return {}; }
        };
        template <class Emit>
        struct PrimitiveSink
        {
            Emit *emit;
            PrimitiveSink &operator*() { return *this; }
            PrimitiveSink &operator++() { return *this; }
            PrimitiveSink operator++(int) { return *this; }
            PrimitiveSink &operator=(BoxIterator item)
            {
                (*emit)(item);
                return *this;
            }
        };
        struct CGALIndex
        {
            // The vector precedes the tree: iterator-backed primitives remain
            // valid through destruction. Data is shared, never moved afterward.
            const std::vector<SpatialBox> boxes;
            CGAL::AABB_tree<BoxTraits> tree;
            explicit CGALIndex(const std::vector<SpatialBox> &input)
                : boxes(input), tree(boxes.begin(), boxes.end())
            {
                tree.build(); // Eager construction; const queries never build.
            }
        };
    }
    struct BoundingBoxTree::Data
    {
        struct Node
        {
            SpatialBox box;
            index left = -1, right = -1, item = -1;
        };
        BoundingBoxBackend backend;
        index size, used = 0;
        std::vector<Node> nodes;
        std::unique_ptr<const CGALIndex> cgal;
        Data(const std::vector<SpatialBox> &boxes, BoundingBoxBackend selected)
            : backend(selected), size(index(boxes.size()))
        {
            DNDS_check_throw_info(selected == BoundingBoxBackend::Native || selected == BoundingBoxBackend::CGAL, "unknown bounding-box backend");
            for (const auto &box : boxes)
                DNDS_check_throw_info(box.lower.allFinite() && box.upper.allFinite() &&
                                          (box.lower.array() <= box.upper.array()).all(),
                                      "invalid bounding box");
            if (selected == BoundingBoxBackend::CGAL)
                cgal = std::make_unique<const CGALIndex>(boxes);
            else
            {
                nodes.resize(boxes.empty() ? 0 : 2 * boxes.size() - 1);
                std::vector<index> order(boxes.size());
                std::iota(order.begin(), order.end(), index(0));
                if (!boxes.empty())
                    Build(order, 0, size, boxes);
            }
        }
        index Build(std::vector<index> &order, index first, index end, const std::vector<SpatialBox> &boxes)
        {
            const index node = used++;
            for (index i = first; i < end; ++i)
                nodes[node].box.Add(boxes[order[i]]);
            if (end - first == 1)
                nodes[node].item = order[first];
            else
            {
                int axis = 0;
                (nodes[node].box.upper - nodes[node].box.lower).maxCoeff(&axis);
                const index middle = first + (end - first) / 2;
                std::nth_element(order.begin() + first, order.begin() + middle, order.begin() + end, [&](index a, index b)
                                 {
                    const real ca = 0.5 * boxes[a].lower[axis] + 0.5 * boxes[a].upper[axis];
                    const real cb = 0.5 * boxes[b].lower[axis] + 0.5 * boxes[b].upper[axis];
                    return ca == cb ? a < b : ca < cb; });
                nodes[node].left = Build(order, first, middle, boxes);
                nodes[node].right = Build(order, middle, end, boxes);
            }
            return node;
        }
    };
    BoundingBoxTree::BoundingBoxTree(const std::vector<SpatialBox> &boxes, BoundingBoxBackend backend)
        : _data(std::make_shared<const Data>(boxes, backend)) {}
    index BoundingBoxTree::Size() const { return _data->size; }
    BoundingBoxBackend BoundingBoxTree::Backend() const { return _data->backend; }
    void BoundingBoxTree::QueryImpl(const tPoint &point, real gap, void *context, Visitor visit) const
    {
        // Pin the index for the complete synchronous visit, even if a callback
        // replaces the caller's last handle. No ownership work occurs per hit.
        const auto data = _data;
        DNDS_check_throw_info(data && point.allFinite() && std::isfinite(gap) && gap >= 0, "invalid bounding-box query or moved index");
        if (!data->size)
            return;
        const real squaredGap = gap * gap;
        if (data->backend == BoundingBoxBackend::Native)
        {
            // Median splitting bounds depth by signed-index bits.
            std::array<index, 128> stack{};
            int size = 1;
            while (size)
            {
                const auto &node = data->nodes[stack[--size]];
                if (node.box.SquaredDistance(point) > squaredGap)
                    continue;
                if (node.item >= 0)
                    visit(context, node.item);
                else
                {
                    DNDS_assert(size + 2 <= int(stack.size()));
                    stack[size++] = node.left;
                    stack[size++] = node.right;
                }
            }
            return;
        }
        const auto &cgal = *data->cgal;
        auto emit = [&](BoxIterator item)
        {
            // Cuboid broad phase includes corners outside the Euclidean gap.
            if (item->SquaredDistance(point) <= squaredGap)
                visit(context, index(item - cgal.boxes.begin()));
        };
        // Enclose every candidate accepted by the floating squared predicate,
        // including subnormal squared gaps and zero-gap distance underflow.
        // Outward rounding prevents a cuboid from losing a tangent candidate.
        const real infinity = std::numeric_limits<real>::infinity();
        const real limit = std::numeric_limits<real>::max();
        const real radius = std::max(gap, std::sqrt(std::nextafter(squaredGap, infinity)));
        tPoint lower, upper;
        for (int d = 0; d < 3; ++d)
        {
            lower[d] = std::clamp(std::nextafter(point[d] - radius, -infinity), -limit, limit);
            upper[d] = std::clamp(std::nextafter(point[d] + radius, infinity), -limit, limit);
        }
        const Kernel::Iso_cuboid_3 query({lower[0], lower[1], lower[2]}, {upper[0], upper[1], upper[2]});
        cgal.tree.all_intersected_primitives(query, PrimitiveSink<decltype(emit)>{&emit});
    }
}
