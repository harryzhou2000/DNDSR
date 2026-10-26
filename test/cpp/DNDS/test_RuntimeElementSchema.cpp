#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "DNDS/Runtime/ElementSchema.hpp"
#include "DNDS/Runtime/RowTransfer.hpp"
#include "Geom/Mesh/MeshTypes.hpp"
#include "Geom/PeriodicTypes.hpp"
#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>

namespace R = DNDS::Runtime;

namespace
{
    struct alignas(32) Mixed
    {
        uint8_t tag;
        int16_t small;
        int32_t code;
        int64_t id;
        double weight;
        int16_t lanes[3];
        uint32_t flags[2];
    };

    R::ElementSchema MixedSchema(size_t alignment = 1, std::string identity = "test.Mixed.v1")
    {
        return R::ElementSchema::Record(std::move(identity), sizeof(Mixed),
                                        {{"tag", offsetof(Mixed, tag), R::ElementScalar::U8},
                                         {"small", offsetof(Mixed, small), R::ElementScalar::I16},
                                         {"code", offsetof(Mixed, code), R::ElementScalar::I32},
                                         {"id", offsetof(Mixed, id), R::ElementScalar::I64},
                                         {"weight", offsetof(Mixed, weight), R::ElementScalar::F64},
                                         {"lanes", offsetof(Mixed, lanes), R::ElementScalar::I16, 3},
                                         {"flags", offsetof(Mixed, flags), R::ElementScalar::U32, 2}},
                                        "Mixed", alignment);
    }

    bool g_preMPIRegistration = false, g_preMPITypeRejected = false;
    std::optional<R::ElementSchema> g_preMPISchema;
}

TEST_CASE("ElementSchema registration is MPI-free and shares canonical native definitions")
{
    REQUIRE(g_preMPIRegistration);
    CHECK(g_preMPITypeRejected);
    const auto schema = R::ElementSchema::For<Mixed>();
    CHECK(schema.SharesDefinition(*g_preMPISchema));
    CHECK(schema.SharesDefinition(R::ElementSchema::Register<Mixed>(MixedSchema())));
    CHECK(schema.Alignment() == alignof(Mixed));
    CHECK(schema.Extent() == sizeof(Mixed));
    CHECK(schema.Identity() == "test.Mixed.v1");
    CHECK(schema.CppTypeName() == "Mixed");
    CHECK(schema.Compatible(R::ElementSchema::TypeIdentity<Mixed>()));
    CHECK(schema.BindingIdentity() == R::ElementSchema::TypeIdentity<Mixed>());
    CHECK_FALSE(schema.Compatible(R::ElementSchema::TypeIdentity<double>()));
    auto reverseFields = schema.Fields();
    std::reverse(reverseFields.begin(), reverseFields.end());
    const auto differentlyNamed = R::ElementSchema::Record("other.logical.identity", sizeof(Mixed), reverseFields,
                                                           "DiagnosticNameOnly", alignof(Mixed));
    CHECK(differentlyNamed.Definition() == schema.Definition());
    CHECK(differentlyNamed.Identity() != schema.Identity());
    const auto representation = schema.WithRepresentation("numpy.Mixed");
    CHECK(representation.SharesDefinition(schema));
    CHECK(representation.Definition() == schema.Definition());
    CHECK(representation.CompatibleRepresentation("numpy.Mixed"));
    CHECK_FALSE(representation.CompatibleRepresentation("numpy.Other"));
    CHECK(representation.Compatible(R::ElementSchema::TypeIdentity<Mixed>()));
    CHECK(representation.BindingIdentity() == schema.BindingIdentity());
    CHECK(R::ElementSchema::For<double>().SharesDefinition(R::ElementSchema::Arithmetic<double>()));
    CHECK(R::ElementSchema::TypeIdentity<bool>() == "bool");
    CHECK(R::ElementSchema::For<bool>().Identity() == "bool");
    CHECK(R::ElementSchema::For<bool>().BindingIdentity() == "bool");
    CHECK(R::ElementSchema::For<bool>().SharesDefinition(R::ElementSchema::Arithmetic<bool>()));
    CHECK(R::ElementSchema::For<DNDS::Geom::ElemInfo>().Identity() == "Geom.ElemInfo.v1");
    CHECK(R::ElementSchema::For<DNDS::Geom::NodePeriodicBits>().Identity() == "Geom.NodePeriodicBits.v1");
}

TEST_CASE("ElementSchema swaps each mixed scalar word and preserves all padding")
{
    const auto schema = R::ElementSchema::For<Mixed>();
    std::array<unsigned char, sizeof(Mixed) * 2> bytes{}, expected{};
    for (size_t i = 0; i < bytes.size(); ++i)
        bytes[i] = static_cast<unsigned char>((i * 37 + 11) % 256);
    const auto original = bytes;
    expected = bytes;
    const std::pair<size_t, size_t> words[] = {
        {offsetof(Mixed, tag), 1}, {offsetof(Mixed, small), 2}, {offsetof(Mixed, code), 4}, {offsetof(Mixed, id), 8}, {offsetof(Mixed, weight), 8}, {offsetof(Mixed, lanes), 2}, {offsetof(Mixed, lanes) + 2, 2}, {offsetof(Mixed, lanes) + 4, 2}, {offsetof(Mixed, flags), 4}, {offsetof(Mixed, flags) + 4, 4}};
    std::array<bool, sizeof(Mixed)> declared{};
    for (const auto &[offset, width] : words)
    {
        for (size_t byte = 0; byte < width; ++byte)
            declared[offset + byte] = true;
        for (size_t row = 0; row < 2; ++row)
            for (size_t byte = 0; byte < width; ++byte)
                expected[row * sizeof(Mixed) + offset + byte] = original[row * sizeof(Mixed) + offset + width - 1 - byte];
    }
    schema.SwapWords(bytes.data(), 2);
    CHECK(bytes == expected);
    for (size_t i = 0; i < bytes.size(); ++i)
        if (!declared[i % sizeof(Mixed)])
            CHECK(bytes[i] == original[i]);
    schema.SwapWords(bytes.data(), 2);
    CHECK(bytes == original);
    CHECK_NOTHROW(schema.SwapWords(nullptr, 0));
    CHECK_THROWS(schema.SwapWords(nullptr, 1));
    CHECK_THROWS(schema.SwapWords(bytes.data(), std::numeric_limits<size_t>::max()));
    CHECK(std::count(declared.begin(), declared.end(), false) > 0);
}

TEST_CASE("ElementSchema rejects invalid field layouts and incompatible registrations")
{
    using S = R::ElementSchema;
    using W = R::ElementScalar;
    CHECK_THROWS(S::Record("", 8, {{"x", 0, W::I64}}));
    CHECK_THROWS(S::Record("empty", 8, {}));
    CHECK_THROWS(S::Record("zero", 0, {{"x", 0, W::I8}}));
    CHECK_THROWS(S::Record("overlap", 8, {{"a", 0, W::I32}, {"b", 2, W::I32}}));
    CHECK_THROWS(S::Record("duplicate", 8, {{"a", 0, W::I32}, {"a", 4, W::I32}}));
    CHECK_THROWS(S::Record("outside", 8, {{"x", 8, W::I8}}));
    CHECK_THROWS(S::Record("overflow", 8, {{"x", 1, W::I64, std::numeric_limits<size_t>::max()}}));
    CHECK_THROWS(S::Record("zero-count", 8, {{"x", 0, W::I8, 0}}));
    CHECK_THROWS(S::Record("bad-word", 8, {{"x", 0, static_cast<W>(999)}}));
    CHECK_THROWS(S::Record("bad-alignment", 8, {{"x", 0, W::I32}}, "", 3));
    CHECK_THROWS(S::ScalarBytes(static_cast<W>(999)));
    CHECK_THROWS(S::ScalarName(static_cast<W>(999)));
    CHECK_THROWS(S::Register<Mixed>(MixedSchema(alignof(Mixed), "changed.logical.identity")));
    CHECK_THROWS(S::Register<Mixed>(S::Record("wrong-size", 1, {{"x", 0, W::U8}})));
    CHECK_THROWS(S::Register<Mixed>(MixedSchema(alignof(Mixed) * 2)));
    struct alignas(32) ConflictingIdentity
    {
        unsigned char bytes[sizeof(Mixed)];
    };
    CHECK_THROWS(S::Register<ConflictingIdentity>(S::Record("test.Mixed.v1", sizeof(ConflictingIdentity),
                                                            {{"bytes", 0, W::U8, sizeof(Mixed)}})));
    struct Unregistered
    {
        int32_t value;
    };
    CHECK_THROWS(S::For<Unregistered>());
    CHECK_THROWS(S::Register<double>(S::Record("float64", sizeof(double), {{"value", 0, W::I64}}, "", alignof(double))));
    CHECK(S::For<Mixed>().SharesDefinition(*g_preMPISchema));
    CHECK(std::string(S::ScalarName(W::F64)) == "float64");
    CHECK(S::ScalarBytes(W::I16) == 2);
}

TEST_CASE("ElementSchema retains vended MPI structs after caller handles are freed")
{
    int counts[] = {1, 1, 1, 1, 1, 3, 2};
    MPI_Aint offsets[] = {offsetof(Mixed, tag), offsetof(Mixed, small), offsetof(Mixed, code),
                          offsetof(Mixed, id), offsetof(Mixed, weight), offsetof(Mixed, lanes), offsetof(Mixed, flags)};
    MPI_Datatype types[] = {MPI_UINT8_T, MPI_INT16_T, MPI_INT32_T, MPI_INT64_T, MPI_DOUBLE, MPI_INT16_T, MPI_UINT32_T};
    MPI_Datatype raw = MPI_DATATYPE_NULL, resized = MPI_DATATYPE_NULL;
    MPI_Type_create_struct(7, counts, offsets, types, &raw);
    MPI_Type_create_resized(raw, 0, sizeof(Mixed), &resized);
    MPI_Type_commit(&resized);
    const R::ElementSchema legacy(resized, sizeof(Mixed), "test.vended.Mixed");
    MPI_Type_free(&raw);
    MPI_Type_free(&resized);
    REQUIRE(legacy.CanSwapWords());
    CHECK(legacy.Fields().size() == 10);
    const auto native = R::ElementSchema::For<Mixed>();
    MPI_Aint lower = -1, extent = -1;
    MPI_Type_get_extent(legacy.Type(), &lower, &extent);
    CHECK(lower == 0);
    CHECK(extent == MPI_Aint(sizeof(Mixed)));
    int legacySize = 0, nativeSize = 0;
    MPI_Type_size(legacy.Type(), &legacySize);
    MPI_Type_size(native.Type(), &nativeSize);
    CHECK(legacySize == nativeSize);
    CHECK(legacySize == 1 + 2 + 4 + 8 + 8 + 3 * 2 + 2 * 4);
    Mixed source{};
    source.tag = 7;
    source.small = -123;
    source.code = 419;
    source.id = std::numeric_limits<int64_t>::max() - 81;
    source.weight = 7.25;
    source.lanes[0] = -15;
    source.lanes[1] = 16;
    source.lanes[2] = 17;
    source.flags[0] = 0x13579BDF;
    source.flags[1] = 0x2468ACE0;
    Mixed destination;
    std::memset(&destination, 0xA5, sizeof(destination));
    MPI_Sendrecv(&source, 1, legacy.Type(), 0, 12, &destination, 1, native.Type(), 0, 12, MPI_COMM_SELF, MPI_STATUS_IGNORE);
    CHECK(destination.tag == source.tag);
    CHECK(destination.small == source.small);
    CHECK(destination.code == source.code);
    CHECK(destination.id == source.id);
    CHECK(destination.weight == source.weight);
    for (size_t i = 0; i < 3; ++i)
        CHECK(destination.lanes[i] == source.lanes[i]);
    for (size_t i = 0; i < 2; ++i)
        CHECK(destination.flags[i] == source.flags[i]);
    const auto *receivedBytes = reinterpret_cast<const unsigned char *>(&destination);
    for (size_t byte = 0; byte < sizeof(Mixed); ++byte)
    {
        bool isField = false;
        for (const auto &field : native.Fields())
            isField = isField || (byte >= field.offset && byte - field.offset < field.count * R::ElementSchema::ScalarBytes(field.scalar));
        if (!isField)
            CHECK(receivedBytes[byte] == 0xA5);
    }
    std::array<unsigned char, sizeof(Mixed)> bytes{}, reference{};
    std::memcpy(bytes.data(), &source, sizeof(source));
    reference = bytes;
    legacy.SwapWords(bytes.data(), 1);
    native.SwapWords(reference.data(), 1);
    CHECK(bytes == reference);
}

TEST_CASE("ElementSchema marks unflattened MPI typemaps native opaque")
{
    MPI_Datatype vector = MPI_DATATYPE_NULL;
    MPI_Type_vector(2, 1, 2, MPI_INT32_T, &vector);
    MPI_Type_commit(&vector);
    MPI_Aint lower = 0, extent = 0;
    MPI_Type_get_extent(vector, &lower, &extent);
    const R::ElementSchema opaque(vector, size_t(extent), "test.opaque.vector");
    MPI_Type_free(&vector);
    CHECK_FALSE(opaque.CanSwapWords());
    CHECK(opaque.Fields().empty());
    CHECK(opaque.Definition().find("native-opaque") == 0);
    std::array<int32_t, 3> source{11, 17, 23}, destination{-1, -2, -3};
    CHECK_THROWS(opaque.SwapWords(source.data(), 1));
    MPI_Sendrecv(source.data(), 1, opaque.Type(), 0, 13, destination.data(), 1, opaque.Type(), 0, 13, MPI_COMM_SELF, MPI_STATUS_IGNORE);
    CHECK(destination[0] == 11);
    CHECK(destination[1] == -2);
    CHECK(destination[2] == 23);
    const auto extended = R::ElementSchema::Arithmetic<long double>();
    CHECK_FALSE(extended.CanSwapWords());
    CHECK_NOTHROW(extended.Type());
}

TEST_CASE("Registered ElementSchema supports typed transfer and custom reduction")
{
    const auto schema = R::ElementSchema::For<Mixed>();
    R::Array<Mixed> source(R::RowLayout::Dense(2, 1)), destination(R::RowLayout::Dense(1, 1));
    {
        auto write = source.Write();
        write.At(0, 0) = Mixed{};
        write.At(1, 0) = Mixed{};
        write.At(0, 0).code = 4;
        write.At(1, 0).code = 5;
        write.At(0, 0).weight = 2.5;
        write.At(1, 0).weight = 4;
    }
    R::RowTransferPlan route(MPI_COMM_SELF, 2, 1, {{0, 0, 0}, {0, 1, 0}});
    const auto reducer = R::Reducer::Custom<Mixed>(schema, "test.Mixed.sum", [](Mixed &a, const Mixed &b)
                                                   { a.code += b.code; a.weight += b.weight; }, Mixed{}, R::DestinationInit::Identity);
    R::PayloadPlan(route, source.Layout(), destination.Layout(), schema).Bind(source, destination, reducer).Begin().Wait();
    CHECK(destination.Read().At(0, 0).code == 9);
    CHECK(destination.Read().At(0, 0).weight == 6.5);
    CHECK_THROWS(R::Reducer::Custom<double>(schema, "invalid.binding", [](double &a, double b)
                                            { a += b; }, 0.));
}

int main(int argc, char **argv)
{
    try
    {
        // No runtime MPI service may be needed to define/register native PODs
        // or expose Geom element schemas while Python modules initialize.
        g_preMPISchema.emplace(R::ElementSchema::Register<Mixed>(MixedSchema()));
        const auto scalar = R::ElementSchema::Arithmetic<double>();
        const auto metadata = R::ElementSchema::For<DNDS::Geom::ElemInfo>();
        const auto bits = R::ElementSchema::For<DNDS::Geom::NodePeriodicBits>();
        g_preMPIRegistration = scalar.Extent() == sizeof(double) && metadata.Fields().size() == 2 && bits.Fields().size() == 1;
        try
        {
            g_preMPISchema->Type();
        }
        catch (const std::exception &)
        {
            g_preMPITypeRejected = true;
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << "ElementSchema pre-MPI registration failed: " << error.what() << '\n';
    }
    MPI_Init(&argc, &argv);
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    int result = context.run();
    // Exercise destruction of an owned MPI descriptor after MPI finalization.
    std::optional<R::ElementSchema> retained;
    retained.emplace(MPI_DOUBLE, sizeof(double), "test.finalized.retained");
    MPI_Finalize();
    bool rejected = false;
    try
    {
        retained->Type();
    }
    catch (const std::exception &)
    {
        rejected = true;
    }
    retained.reset();
    g_preMPISchema.reset();
    if (!rejected)
    {
        std::cerr << "ElementSchema accepted Type() after MPI finalization\n";
        result = 1;
    }
    return result;
}
