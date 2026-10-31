// Opt-in broad-phase benchmark. Exact element projection/containment is
// deliberately identical downstream work and is excluded from these timings.
#include "Geom/Spatial/BoundingBoxTree.hpp"
#include "DNDS/MPI.hpp"
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits_3.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/version.h>
#include <chrono>
#include <iostream>
#include <malloc.h>
#include <numeric>
#include <random>

namespace G = DNDS::Geom;
using I = DNDS::index;
using K = CGAL::Simple_cartesian<double>;
using Iterator = std::vector<G::SpatialBox>::const_iterator;

namespace
{
    // Inlined controls isolate the compiled wrapper/callback overhead from
    // splitting and tree storage. Production measurements use BackendTree.
    template <bool UseCentroidSplit>
    class FlatReferenceTree
    {
        struct Node
        {
            G::SpatialBox box;
            I left = -1, right = -1, item = -1;
        };
        std::vector<Node> _nodes;
        I _used = 0;
        I Build(std::vector<I> &order, I first, I end, const std::vector<G::SpatialBox> &boxes)
        {
            const I node = _used++;
            G::SpatialBox centers;
            for (I i = first; i < end; ++i)
            {
                _nodes[node].box.Add(boxes[order[i]]);
                if constexpr (UseCentroidSplit)
                    centers.Add(0.5 * (boxes[order[i]].lower + boxes[order[i]].upper));
            }
            if (end - first == 1)
                _nodes[node].item = order[first];
            else
            {
                int axis = 0;
                if constexpr (UseCentroidSplit)
                    (centers.upper - centers.lower).maxCoeff(&axis);
                else
                    (_nodes[node].box.upper - _nodes[node].box.lower).maxCoeff(&axis);
                const I middle = first + (end - first) / 2;
                std::nth_element(order.begin() + first, order.begin() + middle, order.begin() + end, [&](I a, I b)
                                 {
                    const double ca = boxes[a].lower[axis] + boxes[a].upper[axis], cb = boxes[b].lower[axis] + boxes[b].upper[axis];
                    return ca == cb ? a < b : ca < cb; });
                _nodes[node].left = Build(order, first, middle, boxes);
                _nodes[node].right = Build(order, middle, end, boxes);
            }
            return node;
        }

    public:
        explicit FlatReferenceTree(const std::vector<G::SpatialBox> &boxes) : _nodes(boxes.empty() ? 0 : 2 * boxes.size() - 1)
        {
            std::vector<I> order(boxes.size());
            std::iota(order.begin(), order.end(), I(0));
            if (!boxes.empty())
                Build(order, 0, I(order.size()), boxes);
        }
        template <class F>
        void Query(const G::tPoint &point, double gap, F &&visit) const
        {
            if (_nodes.empty())
                return;
            std::array<I, 128> stack{};
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
                    stack[size++] = node.left;
                    stack[size++] = node.right;
                }
            }
        }
    };
    using CentroidFlatTree = FlatReferenceTree<true>;
    template <G::BoundingBoxBackend Backend>
    class BackendTree
    {
        G::BoundingBoxTree _tree;

    public:
        explicit BackendTree(const std::vector<G::SpatialBox> &boxes) : _tree(boxes, Backend) {}
        template <class F>
        void Query(const G::tPoint &point, double gap, F &&visit) const { _tree.Query(point, gap, std::forward<F>(visit)); }
    };
    struct BoxPrimitive
    {
        using Id = Iterator;
        using Datum = K::Iso_cuboid_3;
        using Point = K::Point_3;
        Iterator source;
        explicit BoxPrimitive(Iterator iterator) : source(iterator) {}
        Id id() const { return source; }
        Datum datum() const
        {
            return {{source->lower[0], source->lower[1], source->lower[2]},
                    {source->upper[0], source->upper[1], source->upper[2]}};
        }
        Point reference_point() const
        {
            const G::tPoint center = 0.5 * (source->lower + source->upper);
            return {center[0], center[1], center[2]};
        }
    };
    using DefaultTraits = CGAL::AABB_traits_3<K, BoxPrimitive>;
    struct CentroidSplit
    {
        template <class PrimitiveIterator>
        void operator()(PrimitiveIterator first, PrimitiveIterator last, const CGAL::Bbox_3 &) const
        {
            // Use center spread rather than the enclosing primitive-box extent.
            // Long thin boxes can otherwise force repeated splits on an axis
            // with equal centers. Persistent input ordinals break remaining ties.
            G::SpatialBox centers;
            for (auto iterator = first; iterator != last; ++iterator)
                centers.Add(0.5 * (iterator->source->lower + iterator->source->upper));
            int axis = 0;
            (centers.upper - centers.lower).maxCoeff(&axis);
            std::nth_element(first, first + (last - first) / 2, last, [axis](const auto &a, const auto &b)
                             {
                const double ca = a.source->lower[axis] + a.source->upper[axis];
                const double cb = b.source->lower[axis] + b.source->upper[axis];
                return ca == cb ? a.source < b.source : ca < cb; });
        }
    };
    struct BallQuery
    {
        G::tPoint point;
        double squaredGap;
    };
    // A supported CGAL traits customization: use exactly our existing
    // Euclidean point/box test at internal nodes and leaves. No triangulation,
    // primitive duplication or heavyweight exact predicates are needed for a
    // conservative floating-point bounding-box candidate search.
    struct BallTraits : DefaultTraits
    {
        struct Do_intersect
        {
            bool operator()(const BallQuery &query, const CGAL::Bbox_3 &box) const
            {
                G::SpatialBox spatial;
                spatial.lower = {box.xmin(), box.ymin(), box.zmin()};
                spatial.upper = {box.xmax(), box.ymax(), box.zmax()};
                return spatial.SquaredDistance(query.point) <= query.squaredGap;
            }
            bool operator()(const BallQuery &query, const BoxPrimitive &primitive) const
            {
                return primitive.source->SquaredDistance(query.point) <= query.squaredGap;
            }
        };
        Do_intersect do_intersect_object() const { return {}; }
    };
    template <class Emit>
    struct PrimitiveSink
    {
        Emit *emit;
        PrimitiveSink &operator*() { return *this; }
        PrimitiveSink &operator++() { return *this; }
        PrimitiveSink operator++(int) { return *this; }
        PrimitiveSink &operator=(Iterator id)
        {
            (*emit)(id);
            return *this;
        }
    };
    template <class Traits, bool UseCentroidSplit = false>
    class CGALBoxes
    {
        // Match the native constructor's ownership: the caller may destroy
        // its input boxes immediately after construction. Borrowed primitive
        // iterators alone would under-report both retained memory and build.
        std::vector<G::SpatialBox> _boxes;
        Iterator _begin;
        CGAL::AABB_tree<Traits> _tree;

    public:
        explicit CGALBoxes(const std::vector<G::SpatialBox> &boxes)
            : _boxes(boxes), _begin(_boxes.begin()), _tree(_boxes.begin(), _boxes.end())
        {
            if constexpr (UseCentroidSplit)
                _tree.custom_build(Traits().compute_bbox_object(), CentroidSplit{});
            else
                _tree.build();
        }
        template <class F>
        void Query(const G::tPoint &point, double gap, F &&visit) const
        {
            auto emit = [&](Iterator iterator)
            {
                if constexpr (std::is_same_v<Traits, DefaultTraits>)
                {
                    // Cuboid queries are L-infinity broad phase. Filter the
                    // extra corners to match the native Euclidean contract.
                    if (iterator->SquaredDistance(point) > gap * gap)
                        return;
                }
                visit(I(iterator - _begin));
            };
            if constexpr (std::is_same_v<Traits, BallTraits>)
                _tree.all_intersected_primitives(BallQuery{point, gap * gap}, PrimitiveSink<decltype(emit)>{&emit});
            else
            {
                G::tPoint lower, upper;
                for (int d = 0; d < 3; ++d)
                {
                    lower[d] = std::nextafter(point[d] - gap, -std::numeric_limits<double>::infinity());
                    upper[d] = std::nextafter(point[d] + gap, std::numeric_limits<double>::infinity());
                }
                const K::Iso_cuboid_3 query({lower[0], lower[1], lower[2]}, {upper[0], upper[1], upper[2]});
                _tree.all_intersected_primitives(query, PrimitiveSink<decltype(emit)>{&emit});
            }
        }
    };
    struct Input
    {
        std::vector<G::SpatialBox> boxes;
        std::vector<G::tPoint> queries;
        double gap;
    };
    Input Generate(const std::string &scenario, I count, I queries)
    {
        Input input;
        input.boxes.resize(size_t(count));
        input.queries.resize(size_t(queries));
        std::mt19937_64 generator(472561);
        std::uniform_real_distribution<double> uniform(0, 1);
        const I side = I(std::ceil(scenario == "surface" || scenario == "shell" ? std::sqrt(double(count)) : std::cbrt(double(count))));
        const double h = 1. / side;
        input.gap = 0.05 * h;
        for (I iBox = 0; iBox < count; ++iBox)
        {
            auto &box = input.boxes[size_t(iBox)];
            if (scenario == "surface" || scenario == "shell")
            {
                for (int j = 0; j < 2; ++j)
                    for (int k = 0; k < 2; ++k)
                    {
                        const double a = (iBox % side + j) * h, b = (iBox / side + k) * h;
                        if (scenario == "surface")
                            box.Add(G::tPoint(a, b, 0.1 * std::sin(6 * a) * std::cos(6 * b)));
                        else
                        {
                            const double theta = 2 * DNDS::pi * a, phi = DNDS::pi * b;
                            box.Add(G::tPoint(std::sin(phi) * std::cos(theta), std::sin(phi) * std::sin(theta), std::cos(phi)));
                        }
                    }
            }
            else
            {
                G::tPoint center((iBox % side + 0.5) * h, ((iBox / side) % side + 0.5) * h, (iBox / (side * side) + 0.5) * h);
                G::tPoint radius = G::tPoint::Constant(0.49 * h);
                if (scenario == "anisotropic")
                    radius = {4 * h, 0.25 * h, 0.08 * h};
                if (scenario == "clustered")
                {
                    const int cluster = int(iBox % 8);
                    center = {0.6 * (cluster % 2) + 0.15 * uniform(generator),
                              0.6 * ((cluster / 2) % 2) + 0.15 * uniform(generator),
                              0.6 * (cluster / 4) + 0.15 * uniform(generator)};
                    radius *= 0.1 + 2 * uniform(generator);
                }
                box.lower = center - radius;
                box.upper = center + radius;
            }
        }
        for (I iQuery = 0; iQuery < queries; ++iQuery)
        {
            if (iQuery % 4 < 2)
            {
                const auto &box = input.boxes[size_t(generator() % count)];
                const G::tPoint random{uniform(generator), uniform(generator), uniform(generator)};
                input.queries[size_t(iQuery)] = box.lower + (box.upper - box.lower).cwiseProduct(random);
            }
            else
            {
                input.queries[size_t(iQuery)] = {uniform(generator), uniform(generator), uniform(generator)};
                if (scenario == "surface")
                    input.queries[size_t(iQuery)][2] = iQuery % 4 == 2 ? 0.1 * std::sin(6 * input.queries[size_t(iQuery)][0]) * std::cos(6 * input.queries[size_t(iQuery)][1]) : 2;
                else if (iQuery % 4 == 3)
                    input.queries[size_t(iQuery)].array() += 3;
            }
        }
        return input;
    }
    struct Digest
    {
        uint64_t hits = 0, sum = 0, hash = 0;
        void Add(I query, I id)
        {
            uint64_t word = uint64_t(id) + 0x9e3779b97f4a7c15ULL * uint64_t(query + 1);
            word = (word ^ (word >> 30)) * 0xbf58476d1ce4e5b9ULL;
            word = (word ^ (word >> 27)) * 0x94d049bb133111ebULL;
            ++hits;
            sum += word;
            hash ^= word ^ (word >> 31);
        }
        bool operator==(const Digest &other) const { return hits == other.hits && sum == other.sum && hash == other.hash; }
    };
    size_t LiveHeap()
    {
        const auto info = mallinfo2();
        return info.uordblks + info.hblkhd;
    }
    template <class Tree>
    Digest Measure(const std::string &scenario, const std::string &backend, const Input &input, int repeat)
    {
        using Clock = std::chrono::steady_clock;
        const auto memoryBefore = LiveHeap();
        const auto begin = Clock::now();
        Tree tree(input.boxes);
        const auto built = Clock::now();
        const auto retained = LiveHeap() - memoryBefore;
        Digest warmup;
        for (I iQuery = 0; iQuery < std::min(I(4096), I(input.queries.size())); ++iQuery)
            tree.Query(input.queries[size_t(iQuery)], input.gap, [&](I id)
                       { warmup.Add(iQuery, id); });
        Digest result;
        const auto started = Clock::now();
        for (I iQuery = 0; iQuery < I(input.queries.size()); ++iQuery)
            tree.Query(input.queries[size_t(iQuery)], input.gap, [&](I id)
                       { result.Add(iQuery, id); });
        const auto finished = Clock::now();
        // Independent exhaustive candidate oracle on sampled queries. No
        // checksum substitutes for set comparison here. Outside timing.
        for (I iQuery = 0; iQuery < std::min(I(64), I(input.queries.size())); ++iQuery)
        {
            std::vector<I> actual, expected;
            const auto &point = input.queries[size_t(iQuery)];
            tree.Query(point, input.gap, [&](I id)
                       { actual.push_back(id); });
            for (I iBox = 0; iBox < I(input.boxes.size()); ++iBox)
                if (input.boxes[size_t(iBox)].SquaredDistance(point) <= input.gap * input.gap)
                    expected.push_back(iBox);
            std::sort(actual.begin(), actual.end());
            DNDS_check_throw_info(actual == expected, "BVH differs from exhaustive oracle");
        }
        std::cout << scenario << ',' << input.boxes.size() << ',' << input.queries.size() << ',' << repeat << ',' << backend << ','
                  << std::chrono::duration<double>(built - begin).count() << ',' << std::chrono::duration<double>(finished - started).count() << ','
                  << retained << ',' << result.hits << ',' << result.sum << ',' << result.hash << std::endl;
        return result;
    }
    template <class Tree>
    void CheckBoundaryCandidates()
    {
        G::SpatialBox line, point, box;
        line.lower = {0, 0, 0};
        line.upper = {1, 0, 0};
        point.Add(G::tPoint(2, 0, 0));
        box.lower = {-1, -1, -1};
        box.upper = {0, 0, 0};
        const std::vector<G::SpatialBox> boxes{line, point, box};
        Tree tree(boxes);
        for (double gap : {0., 0.1})
            for (const G::tPoint &query : {G::tPoint(0, 0, 0), G::tPoint(1, gap, 0), G::tPoint(2, gap, 0),
                                           G::tPoint(1, std::nextafter(gap, 1.), 0), G::tPoint(0.05, 0.05, 0)})
            {
                std::vector<I> actual, expected;
                tree.Query(query, gap, [&](I id)
                           { actual.push_back(id); });
                for (I iBox = 0; iBox < I(boxes.size()); ++iBox)
                    if (boxes[iBox].SquaredDistance(query) <= gap * gap)
                        expected.push_back(iBox);
                std::sort(actual.begin(), actual.end());
                DNDS_check_throw_info(actual == expected, "BVH boundary candidates differ from exhaustive oracle");
            }
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const I count = argc > 1 ? std::stoll(argv[1]) : 262144;
    const I queries = argc > 2 ? std::stoll(argv[2]) : 500000;
    const int repeats = argc > 3 ? std::stoi(argv[3]) : 5;
    const bool centroidPair = argc > 4 && std::string(argv[4]) == "pair";
    DNDS_check_throw_info(count > 0 && queries > 0 && repeats > 0, "positive benchmark sizes required");
    CheckBoundaryCandidates<G::BoundingBoxTree>();
    CheckBoundaryCandidates<BackendTree<G::BoundingBoxBackend::Native>>();
    CheckBoundaryCandidates<CentroidFlatTree>();
    CheckBoundaryCandidates<CGALBoxes<DefaultTraits>>();
    CheckBoundaryCandidates<CGALBoxes<DefaultTraits, true>>();
    CheckBoundaryCandidates<CGALBoxes<BallTraits>>();
    std::cerr << "CGAL " << CGAL_VERSION_STR << "; one thread; direct box primitive IDs; no distance KD tree\n";
    std::cout << "scenario,boxes,queries,repeat,backend,build_s,query_s,retained_heap_bytes,hits,sum,hash\n";
    for (const std::string scenario : {"surface", "shell", "volume", "anisotropic", "clustered"})
    {
        auto input = Generate(scenario, count, queries);
        for (int repeat = 0; repeat < repeats; ++repeat)
        {
            std::optional<Digest> expected;
            // Rotate execution order to reduce systematic cache/thermal bias.
            const int backends = centroidPair ? 4 : 7;
            for (int order = 0; order < backends; ++order)
            {
                Digest result;
                const std::array<int, 4> paired{0, 6, 5, 3};
                const int selected = centroidPair ? paired[(repeat + order) % 4] : (repeat + order) % 7;
                switch (selected)
                {
                case 0:
                    result = Measure<BackendTree<G::BoundingBoxBackend::Native>>(scenario, "native_wrapper", input, repeat);
                    break;
                case 1:
                    result = Measure<CGALBoxes<DefaultTraits>>(scenario, "cgal_cuboid", input, repeat);
                    break;
                case 2:
                    result = Measure<CGALBoxes<BallTraits>>(scenario, "cgal_ball", input, repeat);
                    break;
                case 3:
                    result = Measure<CGALBoxes<DefaultTraits, true>>(scenario, "cgal_inline", input, repeat);
                    break;
                case 4:
                    result = Measure<CentroidFlatTree>(scenario, "native_centroid", input, repeat);
                    break;
                case 5:
                    result = Measure<FlatReferenceTree<false>>(scenario, "native_inline", input, repeat);
                    break;
                default:
                    result = Measure<BackendTree<G::BoundingBoxBackend::CGAL>>(scenario, "cgal_wrapper", input, repeat);
                    break;
                }
                if (expected)
                    DNDS_check_throw_info(result == *expected, "candidate digests differ between BVH backends");
                else
                    expected = result;
            }
        }
    }
    MPI_Finalize();
}
