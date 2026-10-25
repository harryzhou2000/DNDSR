#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "DNDS/Runtime/Array.hpp"
#include <map>

namespace R = DNDS::Runtime;

TEST_CASE("Array allocation accounting retains capacity and deduplicates aliased metadata")
{
    auto backing = R::Buffer::Allocate(128);
    auto first = R::Array<double>::FromStorage(R::RowLayout::Dense(2, 2), backing.Subspan(16, 4 * sizeof(double)));
    auto second = R::Array<double>::FromStorage(R::RowLayout::Dense(1, 1), backing.Subspan(32, sizeof(double)));
    R::Array<double> csr(R::RowLayout::Ragged({2, 0, 3}));
    auto otherValues = R::Array<double>::FromSharedStorage(csr.SharedStorage()->WithValues(R::Buffer::Allocate(5 * sizeof(double)), sizeof(double)));
    R::Array<int> padded(R::RowLayout::Padded({1, 3}, 5));
    std::map<uint64_t, size_t> allocations;
    auto visit = [&](uint64_t identity, size_t capacity, R::Device device)
    {
        CHECK(device == R::Device::Host());
        auto [entry, inserted] = allocations.emplace(identity, capacity);
        CHECK(entry->second == capacity);
    };
    first.SharedStorage()->VisitAllocations(visit);
    second.SharedStorage()->VisitAllocations(visit);
    auto retained = csr.Read();
    csr.SharedStorage()->VisitAllocations(visit);
    otherValues.SharedStorage()->VisitAllocations(visit);
    padded.SharedStorage()->VisitAllocations(visit);
    size_t total = 0;
    for (const auto &[identity, bytes] : allocations)
        total += bytes;
    CHECK(allocations.size() == 6);
    CHECK(total == 128 + 2 * 5 * sizeof(double) + 4 * sizeof(size_t) + 2 * 5 * sizeof(int) + 2 * sizeof(size_t));
}

TEST_CASE("Native failed write scope invalidates its aliases")
{
    R::Array<double> array(R::RowLayout::Dense(1, 1));
    auto alias = array;
    try
    {
        auto write = array.Write();
        write.At(0, 0) = 17;
        throw std::runtime_error("partial operator");
    }
    catch (const std::runtime_error &)
    {
    }
    CHECK_FALSE(alias.IsValid());
    CHECK_THROWS(alias.Read());
    array.Reset(R::RowLayout::Dense(1, 1));
    CHECK(array.IsValid());
    CHECK_FALSE(alias.IsValid());
}

TEST_CASE("native Array retains row layout and allocation across reset")
{
    R::Array<double> a(R::RowLayout::Ragged({2, 0, 3}));
    {
        auto write = a.Write();
        write.At(2, 1) = 42;
        CHECK_THROWS(write.At(1, 0));
        CHECK_THROWS(a.Read());
    }
    CHECK(a.Version() == 1);
    auto alias = a;
    auto read = a.Read();
    a.Reset(R::RowLayout::Dense(8, 5, 8));
    CHECK(a.Identity() != alias.Identity());
    CHECK(read.At(2, 1) == 42);
    CHECK(alias.Read().At(2, 1) == 42);
    auto clone = alias.Clone();
    CHECK(clone.Identity() != alias.Identity());
    CHECK(clone.Read().At(2, 1) == 42);
    CHECK(a.Layout().Elements() == 64);
}

TEST_CASE("native ArrayPair core only needs owned access")
{
    R::ArrayPair<double> pair;
    pair.owned.Reset(R::RowLayout::Dense(4, 2));
    pair.ghost.Reset(R::RowLayout::Dense(2, 2));
    {
        auto incoming = pair.ghost.Write();
        auto core = pair.ReadOwned();
        auto send = pair.owned.Read();
        CHECK(core.View().Rows() == 4);
        CHECK_THROWS(pair.ReadCombined());
        CHECK_THROWS(pair.Read());
        incoming.At(0, 1) = 7;
    }
    auto ring = pair.ReadCombined();
    CHECK(ring.View().Rows() == 6);
    CHECK(ring.View()[4].size() == 2);
    CHECK(ring.View().Row(4)[1] == 7);
    R::Array<double, R::FixedRows<5>> fixed;
    CHECK(fixed.Layout().Width() == 5);
    CHECK_THROWS(fixed.Reset(R::RowLayout::Dense(2, 3)));
}

TEST_CASE("Native sized row views preserve logical extents and retained metadata")
{
    static_assert(std::is_trivially_copyable_v<R::RowSpan<int>>);
    static_assert(std::is_same_v<decltype(std::declval<R::ArrayView<const int>>()[0][0]), const int &>);
    static_assert(std::is_same_v<decltype(std::declval<R::ArrayView<int>>()[0][0]), int &>);
    R::RowSpan<int> empty;
    CHECK(empty.empty());
    CHECK(empty.begin() == empty.end());
    CHECK(empty.cbegin() == empty.cend());

    for (const auto &shape : {R::RowLayout::Dense(3, 2, 4), R::RowLayout::Padded({2, 0, 3}, 4), R::RowLayout::Ragged({2, 0, 3})})
    {
        R::Array<int> values(shape);
        {
            auto write = values.Write();
            auto view = write.View<true>();
            for (size_t row = 0; row < view.Rows(); ++row)
            {
                auto span = view[row];
                CHECK(span.size() == shape.RowSize(row));
                CHECK(span.data() == view.Row(row));
                size_t column = 0;
                for (auto &value : span)
                    value = int(10 * row + column++);
            }
        }
        auto read = values.Read();
        auto view = read.View<true>();
        values.Reset(R::RowLayout::Dense(0, 0));
        for (size_t row = 0; row < view.Rows(); ++row)
        {
            const auto span = view[row];
            CHECK(span.size() == shape.RowSize(row));
            for (size_t column = 0; column < span.size(); ++column)
                CHECK(span[column] == int(10 * row + column));
            CHECK(span.empty() == (span.size() == 0));
        }
    }
}

TEST_CASE("Native fixed POD rows share read and write span access")
{
    struct Entry
    {
        int64_t id;
        double weight;
    };
    R::Array<Entry, R::FixedRows<2>> values(R::RowLayout::Dense(2, 2, 3));
    {
        auto write = values.Write();
        write.View()[1][0] = {71, 1.5};
        write.View()[1][1] = {72, 2.5};
    }
    auto read = values.Read();
    CHECK(read.View()[1].size() == 2);
    CHECK(read.View()[1][0].id == 71);
    CHECK(read.View()[1][1].weight == 2.5);
    CHECK(read.View()[1].data() - read.View()[0].data() == 3);
}

TEST_CASE("native layout zero rows padding and overflow validation")
{
    R::Array<int> zero(R::RowLayout::Ragged({0, 0}));
    CHECK(zero.Read().View().RowSize(1) == 0);
    CHECK_THROWS(R::RowLayout::Dense(1, 4, 2));
    CHECK_THROWS(R::RowLayout::Dense(SIZE_MAX, 2));
    CHECK_THROWS(R::RowLayout::Ragged({SIZE_MAX, 1}));
    R::Array<int> padded(R::RowLayout::Dense(3, 2, 4));
    auto w = padded.Write();
    CHECK(w.View().Row(1) - w.View().Row(0) == 4);
    CHECK_THROWS(w.At(0, 2));
}

TEST_CASE("native typed handles share one allocation and logical metadata record")
{
    R::Array<double, R::FixedRows<5>> fixed(R::RowLayout::Dense(3, 5, 8));
    auto dynamic = R::Array<double>::FromSharedStorage(fixed.SharedStorage());
    CHECK(dynamic.SharedStorage() == fixed.SharedStorage());
    CHECK(dynamic.SharedStorage()->SharedLayout() == fixed.SharedStorage()->SharedLayout());
    dynamic.Write().At(1, 3) = 19;
    CHECK(fixed.Read().At(1, 3) == 19);
    CHECK(fixed.Version() == dynamic.Version());
    {
        auto read = fixed.Read();
        CHECK_THROWS(dynamic.Write());
    }
    CHECK_THROWS((R::Array<double, R::FixedRows<7>>::FromSharedStorage(dynamic.SharedStorage())));
    CHECK_THROWS(R::Array<int32_t>::FromSharedStorage(dynamic.SharedStorage()));
    fixed.Reset(R::RowLayout::Dense(2, 5));
    CHECK(fixed.SharedStorage() != dynamic.SharedStorage());
    CHECK(dynamic.Read().At(1, 3) == 19);

    R::Array<int64_t> ragged(R::RowLayout::Ragged({0, 2, 3}));
    auto alias = R::Array<int64_t>::FromSharedStorage(ragged.SharedStorage());
    CHECK(alias.Layout().Offsets().data() == ragged.Layout().Offsets().data());
    auto access = alias.Write();
    ragged.Reset(R::RowLayout::Dense(0, 0));
    alias.Reset(R::RowLayout::Dense(0, 0));
    access.At(2, 2) = 42; // Lease retains values and metadata after both handles detach.
    CHECK(access.At(2, 2) == 42);
}

TEST_CASE("Imported six-equation state retains external owner and one access domain")
{
    int releases = 0;
    constexpr size_t rows = 37, stride = 8;
    auto owner = std::shared_ptr<void>(new double[rows * stride](), [&releases](void *p)
                                       { ++releases; delete[] static_cast<double *>(p); });
    auto allocation = R::Allocation::Import(owner, rows * stride * sizeof(double));
    auto buffer = R::Buffer::FromAllocation(allocation, 0, allocation.Bytes());
    auto array = R::Array<double, R::FixedRows<6>>::FromStorage(R::RowLayout::Dense(rows, 6, stride), buffer);
    auto ring = buffer.Subspan(30 * stride * sizeof(double), 7 * stride * sizeof(double));
    auto overlap = R::Buffer::FromAllocation(allocation, 31 * stride * sizeof(double), stride * sizeof(double));
    CHECK(ring.Identity() == array.Identity());
    CHECK(overlap.Identity() == array.Identity());
    {
        auto edit = array.Write();
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < 6; ++j)
                edit.At(i, j) = 100 * i + j;
        CHECK_THROWS(ring.Read());
    }
    CHECK(overlap.Read().View<double>()[5] == 3105);
    CHECK(ring.Version() == array.Version());
    CHECK_THROWS(buffer.Subspan(buffer.Bytes(), 1));
    CHECK_THROWS(R::Buffer::FromAllocation(allocation, SIZE_MAX, 1));
    auto misaligned = buffer.Subspan(1, sizeof(double));
    CHECK_THROWS(R::Array<double>::FromStorage(R::RowLayout::Dense(1, 1), misaligned));
    auto lease = array.Read();
    owner.reset();
    allocation = {};
    buffer = ring = overlap = misaligned = {};
    array.Reset(R::RowLayout::Dense(0, 6));
    CHECK(releases == 0);
    CHECK(lease.At(36, 5) == 3605);
    lease.Wait();
    CHECK(releases == 1);
}

TEST_CASE("Imported read-only storage rejects writes without invalidating readers")
{
    auto owner = std::make_shared<std::vector<int64_t>>(19, 41);
    auto token = std::shared_ptr<void>(owner, owner->data());
    auto allocation = R::Allocation::Import(token, owner->size() * sizeof(int64_t), R::Device::Host(), false);
    auto buffer = R::Buffer::FromAllocation(allocation, 0, allocation.Bytes());
    CHECK_THROWS(buffer.Write());
    CHECK(buffer.IsValid());
    CHECK(buffer.Version() == 0);
    CHECK(buffer.Read().View<int64_t>()[18] == 41);
    CHECK_THROWS(R::Allocation::Import({}, 0));
    auto emptyOwner = std::shared_ptr<void>(owner, nullptr);
    CHECK_THROWS(R::Allocation::Import(emptyOwner, 1));
    auto empty = R::Allocation::Import(emptyOwner, 0);
    CHECK(R::Buffer::FromAllocation(empty, 0, 0).Read().Bytes() == 0);
}

TEST_CASE("Independent reconstruction fields share buffered CSR metadata and retain old shape")
{
    std::vector<size_t> lengths;
    for (size_t row = 0; row < 97; ++row)
        lengths.push_back(row % 11 == 0 ? 0 : (row % 3 + 1) * 5);
    auto shape = R::RowLayout::Ragged(lengths);
    auto copiedShape = shape;
    CHECK(copiedShape.Offsets().Identity() == shape.Offsets().Identity());
    R::Array<double> coefficients(shape);
    R::Array<int64_t> indices(shape);
    CHECK(indices.Layout().Offsets().data() == coefficients.Layout().Offsets().data());
    auto secondStorage = coefficients.SharedStorage()->WithValues(R::Buffer::Allocate(shape.Elements() * sizeof(double)), sizeof(double));
    auto second = R::Array<double>::FromSharedStorage(secondStorage);
    CHECK(second.SharedStorage()->SharedLayout() == coefficients.SharedStorage()->SharedLayout());
    CHECK(second.Identity() != coefficients.Identity());
    auto read = coefficients.Read();
    second.Write().At(96, 4) = 17;
    CHECK(read.At(96, 4) == 0);
    coefficients.Reset(R::RowLayout::Dense(1, 5));
    indices.Reset(R::RowLayout::Dense(0, 0));
    copiedShape = shape = {};
    CHECK(read.Layout().RowSize(96) == lengths[96]);
    CHECK(second.Read().At(96, 4) == 17);
    auto metadata = read.Layout().Offsets().Read();
    CHECK(metadata.Bytes() == (lengths.size() + 1) * sizeof(size_t));
}
