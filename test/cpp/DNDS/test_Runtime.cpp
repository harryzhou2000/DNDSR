#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "DNDS/Runtime/Buffer.hpp"
#include "DNDS/Array.hpp"
#include "DNDS/Runtime/Version.hpp"
#include "DNDS/Runtime/Field.hpp"

using namespace DNDS::Runtime;

TEST_CASE("Allocator policy and pooled storage outlive asynchronous access owners")
{
    struct Pool : Allocator
    {
        std::shared_ptr<void> Allocate(size_t bytes, Device device) override
        {
            DNDS_check_throw(device == Device::Host());
            return {new std::byte[bytes](), [](void *p)
                    { delete[] static_cast<std::byte *>(p); }};
        }
    };
    auto pool = std::make_shared<Pool>();
    std::weak_ptr<Pool> alive = pool;
    auto buffer = Buffer::Allocate(16, Device::Host(), pool);
    auto read = buffer.Read();
    pool.reset();
    buffer = {};
    CHECK_FALSE(alive.expired());
    read.Wait();
    CHECK(alive.expired());
}

TEST_CASE("Field keeps association and distribution outside allocation")
{
    Field<double> a({3, 5, 8}, {42, 7, 2, 1}, "cell");
    Field<double> b({3, 5, 8}, {42, 7, 2, 1}, "cell");
    a.Write().View<double>()[8] = 2;
    b.CopyFrom(a).Wait();
    CHECK(b.Read().View<double>()[8] == 2);
    Field<double> other({3, 5, 8}, {42, 8, 2, 1}, "cell");
    CHECK_THROWS(other.CopyFrom(a));
    CHECK_THROWS(Field<double>({3, 5, 4}, {42, 7, 2, 1}, "cell"));
    CHECK_THROWS(Field<double>({3, 5, 8}, {42, 7, 3, 1}, "cell"));
}

TEST_CASE("Mutation versions invalidate selected dependencies and fail closed")
{
    VersionTracker state;
    auto geometry = state.Stamp(Change::Geometry);
    auto layout = state.Stamp(Change::Layout);
    {
        auto read = state.Read();
        CHECK_FALSE(state.CanMutate());
        CHECK_THROWS(state.BeginMutation(Change::Geometry));
    }
    {
        auto edit = state.BeginMutation(Change::Layout);
        CHECK_FALSE(geometry.Valid());
        CHECK_THROWS(state.Read());
        edit.Commit();
    }
    CHECK(geometry.Valid());
    CHECK_FALSE(layout.Valid());
    {
        auto failed = state.BeginMutation(Change::Geometry);
        // An exception/abandoned destructive update must not publish valid caches.
    }
    CHECK_FALSE(geometry.Valid());
    CHECK_FALSE(state.CanMutate());
    CHECK_THROWS(state.Stamp());
}

TEST_CASE("Shallow mesh controller aliases share exclusion and invalidation")
{
    VersionTracker original;
    auto alias = original;
    auto stamp = original.Stamp(Change::Geometry);
    {
        auto reader = alias.Read();
        CHECK_THROWS(original.BeginMutation(Change::Geometry));
    }
    alias.BeginMutation(Change::Geometry).Commit();
    CHECK_FALSE(stamp.Valid());
    VersionTracker replaced;
    auto old = replaced.Stamp();
    replaced = original;
    CHECK_FALSE(old.Valid());
    auto read = replaced.Read();
    CHECK_THROWS(original.BeginMutation(Change::Layout));
    CHECK_THROWS(original.Stamp(Change(128)));
}

TEST_CASE("Runtime aliases share access state and versions, copies do not")
{
    auto a = Buffer::Allocate(4 * sizeof(double));
    auto alias = a;
    auto b = Buffer::Allocate(a.Bytes());
    {
        auto write = a.Write();
        write.View<double>()[2] = 7;
        CHECK_THROWS(alias.Read());
        CHECK_THROWS(alias.Write());
        CHECK(a.Version() == 0);
    }
    CHECK(alias.Version() == 1);
    {
        auto r1 = a.Read();
        auto r2 = alias.Read();
        CHECK(r2.View<double>()[2] == 7);
        CHECK_THROWS(a.Write());
    }
    Copy(a, b).Wait();
    CHECK(b.Identity() != a.Identity());
    CHECK(b.Read().View<double>()[2] == 7);
    CHECK_THROWS(Copy(a, alias));
    CHECK_THROWS(Copy(a, b, ExecutionContext{Device::Host(), reinterpret_cast<void *>(1)}));
}

TEST_CASE("Runtime access owns allocation after handle replacement")
{
    auto a = Buffer::Allocate(2 * sizeof(double));
    auto old = a.Write();
    old.View<double>()[0] = 11;
    a = Buffer::Allocate(sizeof(double));
    CHECK(a.Read().View<double>()[0] == 0);
    CHECK(old.View<double>()[0] == 11);
    old.Wait();
    CHECK_THROWS(old.View<double>());
    CHECK_THROWS(a.Read().View<uint64_t>()[1]);
    auto empty = Buffer::Allocate(0);
    CHECK(empty.Read().View<double>().size == 0);
    auto odd = Buffer::Allocate(3);
    CHECK_THROWS(odd.Read().View<double>());
#ifndef DNDS_USE_CUDA
    CHECK_THROWS(Buffer::Allocate(1, Device::CUDA(0)));
#endif
    CHECK_THROWS(Buffer::Allocate(1, Device{MemorySpace::Host, 0}));
}

TEST_CASE("Legacy Array shares native host allocation without changing row API")
{
    DNDS::Array<double, 3> array;
    array.Resize(2);
    const auto identity = array.HostDataIdentity();
    {
        auto write = array.HostWriteData();
        write.View<double>()[4] = 23;
        CHECK_THROWS(array.HostReadData());
        CHECK_THROWS(array.to_device(DNDS::DeviceBackend::Host));
    }
    CHECK(array(1, 1) == 23);
    CHECK(array.HostDataVersion() == 1);
    auto read = array.HostReadData();
    auto copy = array;
    CHECK(copy.HostDataIdentity() != identity);
    CHECK(copy(1, 1) == 23);
    array.Resize(4);
    CHECK(array.HostDataIdentity() != identity);
    CHECK(read.View<double>()[4] == 23);
}
