#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "DNDS/Core/Algorithm.hpp"
#include "DNDS/Core/FormatSpec.hpp"
#include "DNDS/Core/Offsets.hpp"
#include "DNDS/Core/Permutation.hpp"
#include "DNDS/Core/Ranges.hpp"
#include "DNDS/Algebra/OperandChecks.hpp"
#include "DNDS/Runtime/Array/Array.hpp"

#include <limits>
#include <vector>

TEST_CASE("SortUnique and SortUniqueRemove handle empty, duplicate, reverse, and sentinel rows")
{
    std::vector<DNDS::index> empty;
    DNDS::SortUnique(empty);
    CHECK(empty.empty());

    std::vector<DNDS::index> single{7};
    DNDS::SortUnique(single);
    CHECK(single == std::vector<DNDS::index>{7});

    std::vector<DNDS::index> vals{5, 1, 5, 3, 1, 2, 5};
    DNDS::SortUnique(vals);
    CHECK(vals == std::vector<DNDS::index>{1, 2, 3, 5});

    std::vector<DNDS::index> withSentinel{DNDS::UnInitIndex, 4, DNDS::UnInitIndex, 2, 4, 2, DNDS::UnInitIndex};
    DNDS::SortUniqueRemove(withSentinel, DNDS::UnInitIndex);
    CHECK(withSentinel == std::vector<DNDS::index>{2, 4});

    std::vector<DNDS::index> allSentinel{DNDS::UnInitIndex, DNDS::UnInitIndex};
    DNDS::SortUniqueRemove(allSentinel, DNDS::UnInitIndex);
    CHECK(allSentinel.empty());
}

TEST_CASE("CheckedLowerBoundRow, AllUnique, and IsPermutation cover boundary and error cases")
{
    const std::vector<DNDS::index> empty;
    CHECK_THROWS(DNDS::CheckedLowerBoundRow(empty, DNDS::index(0)));
    CHECK(DNDS::AllUnique(empty));
    CHECK(DNDS::IsPermutation(empty, 0));
    CHECK_FALSE(DNDS::IsPermutation(empty, 1));

    const std::vector<DNDS::index> sorted{2, 5, 9, 14};
    CHECK(DNDS::CheckedLowerBoundRow(sorted, DNDS::index(2)) == 0);
    CHECK(DNDS::CheckedLowerBoundRow(sorted, DNDS::index(14)) == 3);
    CHECK_THROWS(DNDS::CheckedLowerBoundRow(sorted, DNDS::index(1)));
    CHECK_THROWS(DNDS::CheckedLowerBoundRow(sorted, DNDS::index(6)));
    CHECK_THROWS(DNDS::CheckedLowerBoundRow(sorted, DNDS::index(20)));

    CHECK(DNDS::AllUnique(std::vector<DNDS::index>{3, 1, 4, 2}));
    CHECK_FALSE(DNDS::AllUnique(std::vector<DNDS::index>{3, 1, 4, 1}));

    CHECK(DNDS::IsPermutation(std::vector<DNDS::index>{2, 0, 1}, 3));
    CHECK_FALSE(DNDS::IsPermutation(std::vector<DNDS::index>{2, 0, 2}, 3));
    CHECK_FALSE(DNDS::IsPermutation(std::vector<DNDS::index>{0, 1, 3}, 3));
    CHECK_FALSE(DNDS::IsPermutation(std::vector<DNDS::index>{-1, 0, 1}, 3));
}

TEST_CASE("ExclusiveOffsets and BuildCSRFromRows check overflow, negatives, and filtered entries")
{
    CHECK(DNDS::ExclusiveOffsets(std::vector<DNDS::index>{}) == std::vector<DNDS::index>{0});
    CHECK(DNDS::ExclusiveOffsets(std::vector<DNDS::index>{4}) == std::vector<DNDS::index>{0, 4});
    CHECK(DNDS::ExclusiveOffsets(std::vector<DNDS::index>{0, 3, 0, 2}) == std::vector<DNDS::index>{0, 0, 3, 3, 5});
    CHECK_THROWS(DNDS::ExclusiveOffsets(std::vector<DNDS::index>{1, -1, 2}));
    CHECK_THROWS(DNDS::ExclusiveOffsets(std::vector<DNDS::index>{std::numeric_limits<DNDS::index>::max(), 1}));

    const std::vector<std::vector<DNDS::index>> rows{{1, 5, 2}, {}, {0, 3, 7}};
    auto [off, cols] = DNDS::BuildCSRFromRows<int32_t, int32_t>(rows, [](DNDS::index v, int32_t &out)
                                                                {
        if (v < 1 || v > 5)
            return false;
        out = int32_t(v - 1);
        return true; });
    CHECK(off == std::vector<int32_t>{0, 3, 3, 4});
    CHECK(cols == std::vector<int32_t>{0, 4, 1, 2});
}

TEST_CASE("Permutation helpers validate, invert, detect duplicates, and build from ranges")
{
    CHECK_NOTHROW(DNDS::ValidatePermutation(std::vector<DNDS::index>{}));
    CHECK_NOTHROW(DNDS::ValidatePermutation(std::vector<DNDS::index>{2, 0, 1}, 3));
    CHECK_THROWS(DNDS::ValidatePermutation(std::vector<DNDS::index>{2, 0, 1}, 2));
    CHECK_THROWS(DNDS::ValidatePermutation(std::vector<DNDS::index>{0, 0, 1}));
    CHECK_THROWS(DNDS::ValidatePermutation(std::vector<DNDS::index>{0, 3, 1}));

    const auto inv = DNDS::InvertPermutationChecked(std::vector<DNDS::index>{2, 0, 3, 1});
    CHECK(inv == std::vector<DNDS::index>{1, 3, 0, 2});
    CHECK_THROWS(DNDS::InvertPermutationChecked(std::vector<DNDS::index>{1, 1, 0}));

    CHECK_FALSE(DNDS::PermutationHasDuplicates(std::vector<DNDS::index>{4, 1, 2}, 6));
    CHECK(DNDS::PermutationHasDuplicates(std::vector<DNDS::index>{4, 1, 4}, 6));
    CHECK_THROWS(DNDS::PermutationHasDuplicates(std::vector<DNDS::index>{6, 0}, 6));

    auto [order, pos] = DNDS::BuildPermutationFromRanges(4, std::vector<DNDS::index>{2, 0}, std::vector<DNDS::index>{3, 1});
    CHECK(order == std::vector<DNDS::index>{2, 0, 3, 1});
    CHECK(pos == std::vector<DNDS::index>{1, 3, 0, 2});
    CHECK_THROWS(DNDS::BuildPermutationFromRanges(4, std::vector<DNDS::index>{0, 1}, std::vector<DNDS::index>{1, 2}));
    CHECK_THROWS(DNDS::BuildPermutationFromRanges(4, std::vector<DNDS::index>{0, 1}, std::vector<DNDS::index>{2}));
}

TEST_CASE("ValidateContiguousRanges covers empty, unsorted, gap, overlap, and zero-width cases")
{
    CHECK(DNDS::ValidateContiguousRanges({}, 0, false));
    CHECK_FALSE(DNDS::ValidateContiguousRanges({{0, 1}}, 0, false));
    CHECK(DNDS::ValidateContiguousRanges({{2, 5}, {0, 2}, {5, 6}}, 6, false));
    CHECK_FALSE(DNDS::ValidateContiguousRanges({{0, 2}, {3, 5}}, 5, false));
    CHECK_FALSE(DNDS::ValidateContiguousRanges({{0, 3}, {2, 5}}, 5, false));
    CHECK_FALSE(DNDS::ValidateContiguousRanges({{0, 0}, {0, 3}}, 3, false));
    CHECK_THROWS(DNDS::ValidateContiguousRanges({{0, 2}}, 4, true));
}

TEST_CASE("Format::ParseFloatSpec parses alignment, sign, width, precision, and rejects invalid chars")
{
    DNDS::Format::FloatSpec s;
    const std::string specStr = "+08.2e}";
    auto it = DNDS::Format::ParseFloatSpec(specStr.begin(), specStr.end(), s, true);
    CHECK(it == specStr.end() - 1);
    CHECK(s.sign == '+');
    CHECK(s.width == 8);
    CHECK(s.precision == 2);
    CHECK(s.type == 'e');
    const auto formatted = s.ToFormatString(true);
    CHECK((formatted == "{:>+#8.2e}" || formatted == "{:>+8.2e}"));

    DNDS::Format::FloatSpec s2;
    const std::string pStr = ".4g}";
    DNDS::Format::ParseFloatSpec(pStr.begin(), pStr.end(), s2, true);
    CHECK(s2.precision == 4);
    CHECK(s2.type == 'g');
    CHECK(s2.ToFormatString(false) == "{:.4g}");

    const std::string bad = "z}";
    CHECK_THROWS(DNDS::Format::ParseFloatSpec(bad.begin(), bad.end(), s2, true));
}

TEST_CASE("RequirePairwiseDisjoint catches aliased outputs and output-vs-input overlap")
{
    namespace R = DNDS::Runtime;
    R::Array<double> a(R::RowLayout::Dense(4, 2));
    R::Array<double> b(R::RowLayout::Dense(4, 2));
    R::Array<double> c(R::RowLayout::Dense(4, 2));
    CHECK_NOTHROW(DNDS::Algebra::Detail::RequirePairwiseDisjoint({&a, static_cast<const R::Array<double> *>(nullptr)}, {&b, &c}));
    CHECK_THROWS(DNDS::Algebra::Detail::RequirePairwiseDisjoint({&a, &a}, {&b}));
    CHECK_THROWS(DNDS::Algebra::Detail::RequirePairwiseDisjoint({&a}, {&b, &a}));
}
