#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "DNDS/Runtime/RowBuilder.hpp"
#include <thread>

namespace R = DNDS::Runtime;

TEST_CASE("Exclusive padded edits block sharing and retain value addresses across Commit")
{
    R::Array<double> array(R::RowLayout::Padded({5, 0, 7, 2}, 12));
    const auto identity = array.Identity();
    const auto address = array.Storage().LegacyPointer();
    array.Write().At(2, 6) = 19;
    auto edit = array.EditRows();
    CHECK_FALSE(array.IsValid());
    CHECK_THROWS(array.Read());
    CHECK_THROWS(array.Write());
    CHECK_THROWS(array.SharedStorage());
    CHECK_THROWS(R::Array<double>(array));
    bool rejected = false;
    std::thread contender([&]
                          {
        try { auto alias = array; }
        catch (const std::exception &) { rejected = true; } });
    contender.join();
    CHECK(rejected);
    auto &draft = edit.Draft();
    draft.ResizeRow(1, 6);
    draft.Row<double>(1)[5] = 42;
    draft.ResizeRow(2, 4);
    draft.ResizeRow(2, 7);
    CHECK(draft.Row<double>(2)[6] == 0);
    CHECK_THROWS(draft.ResizeRow(0, 13));
    edit.Commit();
    CHECK(array.IsValid());
    CHECK(array.Identity() == identity);
    CHECK(array.Storage().LegacyPointer() == address);
    CHECK(array.ShapeVersion() == 1);
    CHECK(array.Read().At(1, 5) == 42);
    CHECK_THROWS(edit.Commit());
    auto next = array.EditRows();
    next.Draft().ResizeRow(0, 0);
    next.Commit();
    CHECK(array.ShapeVersion() == 2);
    CHECK(array.RowSize(0) == 0);
}

TEST_CASE("Exclusive edits reject aliases pins and independently shared metadata")
{
    R::Array<int64_t> array(R::RowLayout::Ragged({1, 0, 3}));
    {
        auto alias = array;
        CHECK_THROWS(array.EditRows());
    }
    {
        auto metadata = array.Layout();
        CHECK_THROWS(array.EditRows());
    }
    {
        auto read = array.Read();
        CHECK_THROWS(array.EditRows());
        read.Wait();
        CHECK_THROWS(array.EditRows()); // Still pins its layout after access retires.
    }
    {
        auto buffer = array.Storage();
        CHECK_THROWS(array.EditRows());
    }
    {
        auto second = array.SharedStorage()->WithValues(R::Buffer::Allocate(4 * sizeof(int64_t)), sizeof(int64_t));
        CHECK_THROWS(array.EditRows());
    }
    auto edit = array.EditRows();
    edit.Draft().ResizeRow(0, 9);
    edit.Draft().Row<int64_t>(0)[8] = 123;
    edit.Commit();
    CHECK(array.Read().At(0, 8) == 123);
}

TEST_CASE("Abandoned exclusive edits fail closed and table widths survive physical padding")
{
    R::Array<double, R::FixedRows<5>> state(R::RowLayout::Dense(31, 5, 8));
    const auto identity = state.Identity();
    {
        auto edit = state.EditRows();
        CHECK_THROWS(edit.Draft().ResizeRow(0, 8));
        edit.Draft().Row<double>(30)[4] = 99;
        edit.Commit();
    }
    CHECK(state.Identity() == identity);
    CHECK(state.Layout().Stride() == 8);
    CHECK(state.RowSize(30) == 5);
    CHECK(state.Read().At(30, 4) == 99);
    {
        auto abandoned = state.EditRows();
        abandoned.Draft().Row<double>(0)[0] = 7;
    }
    CHECK_FALSE(state.IsValid());
    CHECK_THROWS(state.Read());
    CHECK_THROWS(state.SharedStorage());
    state.Reset(R::RowLayout::Dense(0, 5));
    CHECK(state.IsValid());
    {
        R::Array<double> temporary(R::RowLayout::Ragged({2, 0}));
        auto abandoned = temporary.EditRows();
        temporary.Reset(R::RowLayout::Dense(0, 0));
    } // Editor may be the last owner; releasing it must keep its mutex alive.
}

TEST_CASE("Parallel incremental CSR construction packs once at publication")
{
    constexpr size_t count = 257;
    R::RowBuilder builder(count, sizeof(double));
#pragma omp parallel for
    for (DNDS::index row = 0; row < count; ++row)
    {
        const size_t n = row % 17 == 0 ? 0 : (row % 7 + 1) * 5;
        builder.ResizeRow(row, n / 2);
        auto initial = builder.Row<double>(row);
        for (size_t j = 0; j < initial.size; ++j)
            initial.data[j] = row * 100 + j;
        builder.ResizeRow(row, n); // Grow only this row, retain its prefix.
        auto grown = builder.Row<double>(row);
        for (size_t j = n / 2; j < n; ++j)
            grown.data[j] = row * 100 + j;
    }
    auto array = R::Array<double>::FromSharedStorage(builder.Commit());
    CHECK_FALSE(builder.Active());
    CHECK_THROWS(builder.Commit());
    CHECK_THROWS(builder.ResizeRow(0, 3));
    auto read = array.Read();
    size_t total = 0;
    for (size_t i = 0; i < count; ++i)
    {
        auto n = i % 17 == 0 ? 0 : (i % 7 + 1) * 5;
        CHECK(read.Layout().Offset(i) == total);
        CHECK(read.Layout().RowSize(i) == n);
        for (size_t j = 0; j < n; ++j)
            CHECK(read.At(i, j) == i * 100 + j);
        total += n;
    }
    CHECK(array.Layout().Elements() == total);
}

TEST_CASE("Padded row builder updates logical sizes within fixed capacity")
{
    R::RowBuilder builder(33, sizeof(int64_t), R::RowLayoutKind::Padded, 12);
    for (size_t i = 0; i < builder.Rows(); ++i)
    {
        builder.ResizeRow(i, 12);
        std::fill_n(builder.Row<int64_t>(i).data, 12, int64_t(i + 1));
        builder.ResizeRow(i, i % 11);
    }
    CHECK_THROWS(builder.ResizeRow(32, 13));
    CHECK(builder.RowSize(32) == 10);
    builder.ResizeRow(32, 12);
    CHECK(builder.Row<int64_t>(32)[9] == 33);
    CHECK(builder.Row<int64_t>(32)[10] == 0);
    CHECK(builder.Row<int64_t>(32)[11] == 0);
    auto array = R::Array<int64_t>::FromSharedStorage(builder.Commit());
    CHECK(array.Layout().IsPadded());
    CHECK_FALSE(array.Layout().IsRagged());
    CHECK(array.Layout().Elements() == 33 * 12);
    CHECK(array.Read().View().Row(1) - array.Read().View().Row(0) == 12);
    CHECK_THROWS(array.Read().At(0, 0));
    CHECK_THROWS(R::RowLayout::Padded({0, 13}, 12));
    CHECK_THROWS((R::Array<int64_t, R::FixedRows<12>>::FromSharedStorage(array.SharedStorage())));
}

TEST_CASE("Detached shape drafts preserve old aliases and abandon without publication")
{
    R::Array<double> source(R::RowLayout::Ragged({5, 0, 7}));
    source.Write().At(2, 6) = 99;
    auto retained = source;
    auto read = retained.Read();
    auto draft = R::RowBuilder::FromStorage(*source.SharedStorage());
    draft.ResizeRow(0, 19);
    draft.ResizeRow(1, 6);
    draft.Row<double>(1)[5] = 42;
    CHECK_THROWS(draft.ResizeRow(3, 1));
    CHECK_THROWS(draft.ResizeRow(0, SIZE_MAX));
    CHECK(draft.RowSize(0) == 19);
    source = R::Array<double>::FromSharedStorage(draft.Commit());
    CHECK(source.RowSize(0) == 19);
    CHECK(source.Read().At(1, 5) == 42);
    CHECK(source.Read().At(2, 6) == 99);
    CHECK(retained.RowSize(0) == 5);
    CHECK(read.At(2, 6) == 99);
    auto identity = source.Identity();
    {
        auto abandoned = R::RowBuilder::FromStorage(*source.SharedStorage());
        abandoned.ResizeRow(2, 0);
    }
    CHECK(source.Identity() == identity);
    CHECK(source.Read().At(2, 6) == 99);
    auto write = source.Write();
    CHECK_THROWS(R::RowBuilder::FromStorage(*source.SharedStorage()));
}

TEST_CASE("Builder empty shapes and POD values keep element alignment")
{
    struct Info
    {
        int64_t id;
        double measure;
    };
    R::RowBuilder draft(3, sizeof(Info));
    draft.ResizeRow(2, 2);
    draft.Row<Info>(2)[1] = {51, 0.25};
    auto array = R::Array<Info>::FromSharedStorage(draft.Commit());
    CHECK(array.Read().At(2, 1).id == 51);
    CHECK(array.Read().At(2, 1).measure == 0.25);
    for (auto kind : {R::RowLayoutKind::CSR, R::RowLayoutKind::Padded, R::RowLayoutKind::Table})
    {
        R::RowBuilder empty(0, sizeof(double), kind, 5);
        auto result = R::Array<double>::FromSharedStorage(empty.Commit());
        CHECK(result.Rows() == 0);
        CHECK(result.Layout().Kind() == kind);
    }
    R::RowBuilder widths(3, sizeof(double), R::RowLayoutKind::CSR);
    widths.ResizeRow(1, 5);
    auto pointer = widths.Row<double>(1).data;
    widths.ResizeRow(2, 97); // Another row's growth cannot move this row.
    CHECK(widths.Row<double>(1).data == pointer);
}
