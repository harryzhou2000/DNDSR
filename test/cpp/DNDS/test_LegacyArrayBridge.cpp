#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "DNDS/Runtime/LegacyArrayBridge.hpp"
#include "DNDS/Runtime/RowBuilder.hpp"
#include <array>

namespace R = DNDS::Runtime;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const auto result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}

TEST_CASE("Legacy bridge reuses controller and permits scoped edits both ways")
{
    DNDS::Array<double, 3> legacy;
    legacy.Resize(2);
    legacy(1, 2) = 4;
    auto native = R::ShareLegacyHost(legacy);
    CHECK(native.Identity() == legacy.HostDataIdentity());
    CHECK(native.Storage().LegacyPointer() == legacy.RawDataVector().data());
    CHECK(native.Layout().Width() == 3);
    {
        auto read = legacy.HostReadData();
        CHECK_THROWS(native.Write());
        auto scope = R::HostRowsRead(native);
        CHECK(scope.View().Size() == 2);
        CHECK(scope.View()[1][2] == 4);
        CHECK(scope.View().RowSpan(1).size() == 3);
        CHECK(scope.View().RowSpan(1).data() == scope.View()[1]);
    }
    {
        auto write = native.Write();
        CHECK_THROWS(legacy.HostReadData());
        CHECK_THROWS(legacy.HostWriteData());
        write.View()(1, 2) = 9;
    }
    {
        auto scope = R::HostRowsWrite(legacy);
        CHECK(scope.View()(1, 2) == 9);
        scope.View()(0, 1) = 7;
        scope.View().RowSpan(0)[2] = 8;
        CHECK_THROWS(native.Read());
    }
    CHECK(native.Read().View()(0, 1) == 7);
    CHECK(native.Read().View()[0][2] == 8);
    CHECK(native.Version() == legacy.HostDataVersion());
    CHECK_THROWS(native.EditRows());
    CHECK_THROWS(legacy.RawDataVector().to_device(DNDS::DeviceBackend::Custom1));
    CHECK(native.IsValid());
}

TEST_CASE("Legacy bridge retains exact allocations through replacement and deep copies")
{
    R::Array<double> retained;
    {
        DNDS::Array<double, DNDS::DynamicSize> legacy;
        legacy.Resize(2, 2);
        legacy(0, 0) = 31;
        retained = R::ShareLegacyHost(legacy);
        auto copy = legacy;
        CHECK(copy.HostDataIdentity() != retained.Identity());
        copy(0, 0) = 13;
        CHECK(retained.Read().View()(0, 0) == 31);
        legacy.Resize(4, 3);
        CHECK(legacy.HostDataIdentity() != retained.Identity());
        CHECK(retained.Rows() == 2);
        CHECK(retained.RowSize(0) == 2);
        legacy = copy;
        CHECK(retained.Read().View()(0, 0) == 31);
        auto originalIdentity = retained.Identity();
        copy.SwapData(legacy);
        CHECK(retained.Identity() == originalIdentity);
    }
    CHECK(retained.Read().View()(0, 0) == 31);
    auto old = R::HostRowsRead(retained);
    retained.Reset(R::RowLayout::Dense(1, 1));
    CHECK(old.View().Size() == 2);
    CHECK(old.View()(0, 0) == 31);
    CHECK(old.View().RowSpan(0).size() == 2);
    CHECK(old.View().RowSpan(0)[0] == 31);
}

TEST_CASE("Legacy bridge converts CSR offsets and retains published row metadata")
{
    DNDS::Array<DNDS::index, DNDS::NonUniformSize> legacy;
    legacy.Resize(3, 0);
    CHECK_THROWS(R::ShareLegacyHost(legacy));
    legacy.ResizeRow(0, 2);
    legacy.ResizeRow(2, 1);
    legacy(0, 0) = 10;
    legacy(0, 1) = 20;
    legacy(2, 0) = 30;
    legacy.Compress();
    auto native = R::ShareLegacyHost(legacy);
    CHECK(native.Layout().IsRagged());
    CHECK(native.Layout().Offsets()[1] == 2);
    CHECK(native.Layout().Offsets()[2] == 2);
    CHECK(native.Layout().Offsets()[3] == 3);
    {
        auto write = native.Write();
        CHECK_THROWS(legacy.Decompress());
        CHECK(legacy.IfCompressed());
    }
    legacy.Decompress();
    legacy.ResizeRow(0, 4);
    legacy(0, 0) = 40;
    legacy.Compress();
    CHECK(native.Identity() != legacy.HostDataIdentity());
    CHECK(native.RowSize(0) == 2);
    CHECK(native.Read().View()(0, 0) == 10);
    auto current = R::ShareLegacyHost(legacy);
    CHECK(current.RowSize(0) == 4);
    CHECK(current.Read().View()(0, 0) == 40);
}

template <DNDS::rowsize Capacity>
static void CheckPadded()
{
    DNDS::Array<double, DNDS::NonUniformSize, Capacity> legacy;
    if constexpr (Capacity == DNDS::DynamicSize)
        legacy.Resize(2, 4);
    else
        legacy.Resize(2);
    legacy.ResizeRow(0, 1);
    legacy.ResizeRow(1, 3);
    legacy(1, 2) = 12;
    auto native = R::ShareLegacyHost(legacy);
    CHECK(native.Layout().IsPadded());
    CHECK(native.Layout().Stride() == 4);
    CHECK(native.RowSize(0) == 1);
    CHECK(native.RowSize(1) == 3);
    legacy.ResizeRow(0, 2);
    CHECK(native.RowSize(0) == 1);
    CHECK(R::ShareLegacyHost(legacy).RowSize(0) == 2);
    CHECK(native.Read().View()(1, 2) == 12);
    // Legacy metadata escape remains possible, so publication must validate it.
    legacy.getRowSizes()->at(0) = -1;
    CHECK_THROWS(R::ShareLegacyHost(legacy));
    legacy.getRowSizes()->at(0) = 5;
    CHECK_THROWS(R::ShareLegacyHost(legacy));
    CHECK(native.RowSize(0) == 1);
    CHECK(native.IsValid());
}

TEST_CASE("Legacy bridge snapshots empty publication and escaped padded row sizes")
{
    DNDS::Array<double, 2> fixed;
    DNDS::Array<double, DNDS::DynamicSize> dynamic;
    DNDS::Array<double, DNDS::NonUniformSize, 4> paddedStatic;
    DNDS::Array<double, DNDS::NonUniformSize, DNDS::DynamicSize> paddedDynamic;
    CHECK(R::ShareLegacyHost(fixed).Rows() == 0);
    CHECK(R::ShareLegacyHost(dynamic).Rows() == 0);
    CHECK(R::ShareLegacyHost(paddedStatic).Layout().Stride() == 4);
    CHECK(R::ShareLegacyHost(paddedDynamic).Rows() == 0);
    DNDS::Array<double, DNDS::NonUniformSize> csr;
    CHECK_THROWS(R::HostRowsRead(std::as_const(csr)));
    csr.Compress();
    auto publishedEmpty = R::ShareLegacyHost(csr);
    CHECK(publishedEmpty.Rows() == 0);
    CHECK(publishedEmpty.Layout().Offsets().size() == 1);
    CHECK(publishedEmpty.Layout().Offsets()[0] == 0);
    csr.Decompress();
    CHECK_THROWS(R::ShareLegacyHost(csr));
    CHECK(publishedEmpty.IsValid());

    paddedStatic.Resize(1);
    paddedStatic.ResizeRow(0, 1);
    paddedStatic.RawDataVector()[2] = 17;
    auto before = R::ShareLegacyHost(paddedStatic);
    auto escapedSizes = paddedStatic.getRowSizes();
    escapedSizes->at(0) = 3;
    auto after = R::ShareLegacyHost(paddedStatic);
    CHECK(before.Identity() == after.Identity());
    CHECK(before.RowSize(0) == 1);
    CHECK(after.RowSize(0) == 3);
    CHECK(after.Read().View()(0, 2) == 17);
    auto read = R::HostRowsRead(before);
    CHECK_THROWS(after.Write());
    read.Wait();
    paddedStatic.RawDataVector().clear();
    CHECK_THROWS(R::ShareLegacyHost(paddedStatic));
    CHECK(after.Read().View()(0, 2) == 17);
}

TEST_CASE("Legacy bridge converts static and dynamic padded lengths")
{
    CheckPadded<4>();
    CheckPadded<DNDS::DynamicSize>();
}

TEST_CASE("Legacy bridge validates actual shape byte range type schema and host placement")
{
    DNDS::Array<double, 2> legacy;
    legacy.Resize(2);
    legacy.RawDataVector().resize(1);
    CHECK_THROWS(R::ShareLegacyHost(legacy));
    legacy.Resize(2);
    CHECK_THROWS(R::ShareLegacyHost(legacy, R::Device::CUDA(0)));
    CHECK_THROWS(R::ShareLegacyHost(legacy, R::ElementSchema::Arithmetic<int64_t>()));
    CHECK(R::ShareLegacyHost(legacy, R::ElementSchema::Arithmetic<double>()).Rows() == 2);
    CHECK_THROWS(R::Detail::LegacyExtent(DNDS::index(-1)));
    auto oversized = R::Array<double>(R::RowLayout::Dense(0, size_t(std::numeric_limits<DNDS::rowsize>::max()) + 1));
    CHECK_THROWS(R::HostRowsWrite(oversized));
    CHECK(oversized.IsValid());
    // Empty published layouts still have a real tracked zero-byte allocation.
    DNDS::Array<double, 3> empty;
    CHECK(R::ShareLegacyHost(empty).Rows() == 0);
    static_assert(std::is_trivially_copyable_v<R::LegacyRowsView<const double>>);
}

TEST_CASE("Legacy bridge rejects malformed CSR publication and retains native subspan offsets")
{
    struct Fixture : DNDS::Array<DNDS::index, DNDS::NonUniformSize>
    {
        using DNDS::Array<DNDS::index, DNDS::NonUniformSize>::_pRowStart;
        using DNDS::Array<DNDS::index, DNDS::NonUniformSize>::_size;
    };
    Fixture fixture;
    fixture.Resize(2, [](DNDS::index)
                   { return DNDS::rowsize(1); });
    auto &legacy = static_cast<DNDS::Array<DNDS::index, DNDS::NonUniformSize> &>(fixture);
    auto old = R::ShareLegacyHost(legacy);
    fixture._pRowStart->at(1) = -1;
    CHECK_THROWS(R::ShareLegacyHost(legacy));
    fixture._pRowStart->at(1) = 3;
    CHECK_THROWS(R::ShareLegacyHost(legacy));
    fixture._pRowStart->at(1) = 1;
    fixture._pRowStart->at(2) = 1;
    CHECK_THROWS(R::ShareLegacyHost(legacy));
    fixture._pRowStart->at(2) = 2;
    fixture._pRowStart->at(0) = 1;
    CHECK_THROWS(R::ShareLegacyHost(legacy));
    fixture._pRowStart->at(0) = 0;
    fixture._size = -1;
    CHECK_THROWS(R::ShareLegacyHost(legacy));
    fixture._size = 2;
    CHECK(old.RowSize(0) == 1);
    CHECK(old.RowSize(1) == 1);

    auto allocation = R::Buffer::Allocate(6 * sizeof(double));
    {
        auto write = allocation.Write();
        for (size_t i = 0; i < 6; ++i)
            write.View<double>().data[i] = double(i + 10);
    }
    auto sliced = R::Array<double>::FromStorage(R::RowLayout::Dense(2, 2), allocation.Subspan(sizeof(double), 4 * sizeof(double)));
    {
        auto access = R::HostRowsWrite(sliced);
        CHECK(access.View()[0][0] == 11);
        CHECK(access.View()[1][1] == 14);
        access.View()(1, 1) = 24;
    }
    CHECK(allocation.Read().View<double>().data[4] == 24);
    CHECK(allocation.Read().View<double>().data[5] == 15);
    CHECK_THROWS(R::Array<double>::FromStorage(R::RowLayout::Dense(3, 2), sliced.Storage()));
}

TEST_CASE("Legacy bridge supports a registered POD MPI extent without changing element identity")
{
    struct Pod
    {
        double x;
        int64_t tag;
    };
    static_assert(sizeof(Pod) == 16);
    MPI_Datatype type;
    MPI_Type_contiguous(sizeof(Pod), MPI_BYTE, &type);
    MPI_Type_commit(&type);
    {
        R::ElementSchema schema(type, sizeof(Pod), "bridge.Pod.v1", R::ElementSchema::TypeIdentity<Pod>());
        DNDS::Array<Pod, 1> legacy;
        legacy.Resize(2);
        legacy(1, 0) = {2.5, 19};
        auto native = R::ShareLegacyHost(legacy, schema);
        CHECK(native.Identity() == legacy.HostDataIdentity());
        auto scope = R::HostRowsWrite(native);
        CHECK(scope.View()(1, 0).tag == 19);
        scope.View()(1, 0).x = 8.5;
        scope.Wait();
        CHECK(legacy(1, 0).x == 8.5);
    }
    MPI_Type_free(&type);
}
