#pragma once

#include "Geom/Geometric.hpp"
#include <numeric>

namespace DNDS::Geom
{
    struct SpatialBox
    {
        tPoint lower = tPoint::Constant(veryLargeReal), upper = tPoint::Constant(-veryLargeReal);
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
    };
    /// Local spatial candidate search, independent of mesh entity types. Boxes
    /// and tree nodes are allocated once. Exact acceptance belongs to the caller.
    class BoundingBoxTree
    {
        struct Node
        {
            SpatialBox box;
            index left = -1, right = -1, item = -1;
        };
        std::vector<Node> _nodes;
        index _used = 0;
        index Build(std::vector<index> &order, index first, index end, const std::vector<SpatialBox> &boxes)
        {
            const index node = _used++;
            for (index i = first; i < end; ++i)
                _nodes[node].box.Add(boxes[order[i]]);
            if (end - first == 1)
                _nodes[node].item = order[first];
            else
            {
                int axis = 0;
                (_nodes[node].box.upper - _nodes[node].box.lower).maxCoeff(&axis);
                const index middle = first + (end - first) / 2;
                std::nth_element(order.begin() + first, order.begin() + middle, order.begin() + end,
                                 [&](index a, index b)
                                 {
                                     const real ca = boxes[a].lower[axis] + boxes[a].upper[axis], cb = boxes[b].lower[axis] + boxes[b].upper[axis];
                                     return ca == cb ? a < b : ca < cb;
                                 });
                _nodes[node].left = Build(order, first, middle, boxes);
                _nodes[node].right = Build(order, middle, end, boxes);
            }
            return node;
        }

    public:
        explicit BoundingBoxTree(const std::vector<SpatialBox> &boxes) : _nodes(boxes.empty() ? 0 : 2 * boxes.size() - 1)
        {
            std::vector<index> order(boxes.size());
            std::iota(order.begin(), order.end(), index(0));
            if (!boxes.empty())
                Build(order, 0, index(order.size()), boxes);
        }
        /// Depth-first traversal with a bounded stack: median splits bound depth
        /// by signed-index bits. Callback gets each box within the stated gap.
        template <class F>
        void Query(const tPoint &point, real gap, F &&visit) const
        {
            if (_nodes.empty())
                return;
            std::array<index, 128> stack{};
            int size = 1;
            while (size)
            {
                const auto &node = _nodes[stack[--size]];
                if (node.box.SquaredDistance(point) > gap * gap)
                    continue;
                if (node.item >= 0)
                    visit(node.item);
                else
                {
                    DNDS_assert(size + 2 <= int(stack.size()));
                    stack[size++] = node.left, stack[size++] = node.right;
                }
            }
        }
    };
}
