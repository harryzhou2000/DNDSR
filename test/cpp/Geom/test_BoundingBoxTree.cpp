#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Geom/Spatial/BoundingBoxTree.hpp"
#include <future>
#include <random>

namespace G = DNDS::Geom;
using I = DNDS::index;

namespace
{
    std::vector<I> Query(const G::BoundingBoxTree &tree, const G::tPoint &point, double gap)
    {
        std::vector<I> result;
        tree.Query(point, gap, [&](I iBox)
                   { result.push_back(iBox); });
        std::sort(result.begin(), result.end());
        return result;
    }
    std::vector<I> Oracle(const std::vector<G::SpatialBox> &boxes, const G::tPoint &point, double gap)
    {
        std::vector<I> result;
        for (I iBox = 0; iBox < I(boxes.size()); ++iBox)
            if (boxes[iBox].SquaredDistance(point) <= gap * gap)
                result.push_back(iBox);
        return result;
    }
}

TEST_CASE("Switchable BVH candidates equal exhaustive box scans including thin repeated and tiny boxes")
{
    std::mt19937_64 generator(97151);
    std::uniform_real_distribution<double> uniform(-1, 1);
    std::vector<G::SpatialBox> boxes(512);
    for (I iBox = 0; iBox < I(boxes.size()); ++iBox)
    {
        const G::tPoint center{uniform(generator), uniform(generator), uniform(generator)};
        G::tPoint radius{0.1, 0.03, 0.001};
        if (iBox % 5 == 0)
            radius.setZero();
        else if (iBox % 5 == 1)
            radius[2] = 0;
        boxes[iBox].lower = center - radius;
        boxes[iBox].upper = center + radius;
    }
    // Equal centers exercise deterministic ordinal ties, and singleton
    // point/line/plane boxes must not depend on solid-volume predicates.
    boxes.insert(boxes.end(), 17, boxes[1]);
    for (double x : {0., 1e-162, 2e-162, 2.5e-162, 1e-150})
    {
        G::SpatialBox point;
        point.lower = point.upper = {x, 0, 0};
        boxes.push_back(point);
    }
    const double limit = std::numeric_limits<double>::max();
    for (double x : {-limit, limit})
    {
        G::SpatialBox point;
        point.Add({x, 0, 0});
        CHECK(point.lower[0] == x);
        CHECK(point.upper[0] == x);
        boxes.push_back(point);
    }
    const G::BoundingBoxTree native(boxes, G::BoundingBoxBackend::Native);
    const G::BoundingBoxTree cgal(boxes, G::BoundingBoxBackend::CGAL);
    for (int iQuery = 0; iQuery < 400; ++iQuery)
    {
        const G::tPoint point{uniform(generator), uniform(generator), uniform(generator)};
        const double gap = 0.04 * (iQuery % 4);
        const auto expected = Oracle(boxes, point, gap);
        CHECK(Query(native, point, gap) == expected);
        CHECK(Query(cgal, point, gap) == expected);
    }
    for (double gap : {0., 1e-200, 2e-162, 1e-150, 0.1, limit})
        for (const G::tPoint &point : {G::tPoint::Zero().eval(), boxes[1].lower, boxes[1].upper,
                                       G::tPoint(-limit, 0, 0), G::tPoint(limit, 0, 0),
                                       G::tPoint(boxes[1].upper[0], boxes[1].upper[1] + gap, boxes[1].upper[2])})
        {
            if (!point.allFinite())
                continue;
            const auto expected = Oracle(boxes, point, gap);
            CHECK(Query(native, point, gap) == expected);
            CHECK(Query(cgal, point, gap) == expected);
        }
    CHECK(native.Backend() == G::BoundingBoxBackend::Native);
    CHECK(cgal.Backend() == G::BoundingBoxBackend::CGAL);
    CHECK(native.Size() == I(boxes.size()));
    CHECK(cgal.Size() == native.Size());
}

TEST_CASE("BVH wrappers own inputs share immutable copies and support concurrent const queries")
{
    for (auto backend : {G::BoundingBoxBackend::Native, G::BoundingBoxBackend::CGAL})
    {
        const auto make = [&]
        {
            std::vector<G::SpatialBox> boxes(3);
            for (I iBox = 0; iBox < 3; ++iBox)
            {
                boxes[iBox].lower = {double(iBox), 0, 0};
                boxes[iBox].upper = {double(iBox + 1), 1, 0};
            }
            return G::BoundingBoxTree(boxes, backend);
        };
        auto tree = make(); // Source boxes are already destroyed.
        auto copy = tree;
        tree = G::BoundingBoxTree({}, backend); // Old index lives through copy.
        CHECK(tree.Size() == 0);
        CHECK(Query(copy, {1, 0.5, 0}, 0) == std::vector<I>{0, 1});
        std::vector<std::future<bool>> workers;
        for (int worker = 0; worker < 4; ++worker)
            workers.push_back(std::async(std::launch::async, [copy]
                                         {
                for (int iteration = 0; iteration < 100; ++iteration)
                    if (Query(copy, {1, 0.5, 0}, 0) != std::vector<I>{0, 1})
                        return false;
                return true; }));
        for (auto &worker : workers)
            CHECK(worker.get());
        CHECK_THROWS(copy.Query({0.5, 0.5, 0}, 0, [](I)
                                { throw std::runtime_error("visitor failure"); }));
        CHECK(Query(copy, {0.5, 0.5, 0}, 0) == std::vector<I>{0});
        CHECK(Query(tree, {0.5, 0.5, 0}, 0).empty());
        auto replaceable = make();
        std::vector<I> visited;
        replaceable.Query({1, 0.5, 0}, 1, [&](I iBox)
                          {
            visited.push_back(iBox);
            replaceable = G::BoundingBoxTree({}, backend); });
        std::sort(visited.begin(), visited.end());
        CHECK(visited == std::vector<I>{0, 1, 2});
    }
}

TEST_CASE("BVH rejects invalid boxes backends and query inputs before traversal")
{
    for (auto backend : {G::BoundingBoxBackend::Native, G::BoundingBoxBackend::CGAL})
    {
        G::SpatialBox box;
        CHECK_THROWS(G::BoundingBoxTree({box}, backend)); // Empty sentinel is not an element.
        box.lower = G::tPoint::Zero();
        box.upper = G::tPoint::Ones();
        G::BoundingBoxTree tree({box}, backend);
        CHECK_THROWS(Query(tree, G::tPoint::Zero(), -1));
        CHECK_THROWS(Query(tree, G::tPoint::Zero(), std::numeric_limits<double>::infinity()));
        CHECK_THROWS(Query(tree, G::tPoint::Constant(std::numeric_limits<double>::quiet_NaN()), 0));
        box.lower[0] = std::numeric_limits<double>::quiet_NaN();
        CHECK_THROWS(G::BoundingBoxTree({box}, backend));
    }
    CHECK_THROWS(G::BoundingBoxTree({}, G::BoundingBoxBackend(-1)));
    CHECK(G::BoundingBoxTree({}).Backend() == G::BoundingBoxBackend::CGAL);
}
