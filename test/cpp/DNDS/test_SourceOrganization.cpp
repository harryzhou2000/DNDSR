#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "DNDS/Array.hpp"
#include "DNDS/ArrayPair.hpp"
#include "DNDS/ArrayDOF.hpp"
#include "DNDS/Legacy/Array.hpp"
#include "DNDS/Legacy/ArrayPair.hpp"
#include "DNDS/Legacy/ArrayDOF.hpp"
#include "DNDS/Runtime/Array.hpp"
#include <type_traits>

static_assert(std::is_same_v<DNDS::Array<double, 5>, DNDS::Legacy::Array<double, 5>>);
static_assert(std::is_same_v<DNDS::ParArray<DNDS::index, 2>, DNDS::Legacy::ParArray<DNDS::index, 2>>);
static_assert(std::is_same_v<DNDS::ArrayPair<>, DNDS::Legacy::ArrayPair<>>);
static_assert(std::is_same_v<DNDS::ArrayAdjacencyPair<2>, DNDS::Legacy::ArrayAdjacencyPair<2>>);
static_assert(std::is_same_v<DNDS::ArrayDof<5, 1>, DNDS::Legacy::ArrayDof<5, 1>>);
static_assert(!std::is_same_v<DNDS::Legacy::Array<double>, DNDS::Runtime::Array<double>>);

TEST_CASE("Legacy canonical and forwarding headers preserve the same array type")
{
    DNDS::Legacy::Array<double, 2> values;
    values.Resize(2);
    DNDS::Array<double, 2> &oldSpelling = values;
    oldSpelling(1, 1) = 23;
    CHECK(values(1, 1) == 23);
    DNDS::Runtime::Array<double, DNDS::Runtime::FixedRows<2>> native(DNDS::Runtime::RowLayout::Dense(2, 2));
    native.Write().View()(1, 1) = 47;
    CHECK(native.Read().View()(1, 1) == 47);
    CHECK(oldSpelling(1, 1) == 23);
}
