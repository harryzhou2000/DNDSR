#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "DNDS/Runtime/Array.hpp"

namespace R = DNDS::Runtime;

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
        auto core = pair.owned.Read();
        auto send = pair.owned.Read();
        CHECK(core.View().Rows() == 4);
        CHECK_THROWS(pair.Read());
        incoming.At(0, 1) = 7;
    }
    auto ring = pair.Read();
    CHECK(ring.View().Row(4)[1] == 7);
    R::Array<double, R::FixedRows<5>> fixed;
    CHECK(fixed.Layout().Width() == 5);
    CHECK_THROWS(fixed.Reset(R::RowLayout::Dense(2, 3)));
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
